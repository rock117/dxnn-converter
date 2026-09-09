"""DXNN Converter — FastAPI web service.

Upload a .pt YOLO model, convert it to .dxnn for DEEPX DX-M1 NPU.
Pipeline: .pt → yolo.export(format="deepx") → .dxnn
"""
import importlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# ---- Paths ----
BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "work"
WORK_DIR.mkdir(exist_ok=True)


def _dx_com_available() -> bool:
    """True when the pip-installed dx_com Python package can be imported."""
    try:
        importlib.import_module("dx_com")
        return True
    except ImportError:
        return False


# ---- Task store (in-memory; tasks dir for persistence) ----
TASKS_DIR = BASE_DIR / "tasks"
TASKS_DIR.mkdir(exist_ok=True)

tasks: dict[str, dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _task_dir(task_id: str) -> Path:
    d = WORK_DIR / task_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_task(task_id: str):
    t = tasks.get(task_id)
    if t:
        (TASKS_DIR / f"{task_id}.json").write_text(
            json.dumps(t, ensure_ascii=False, default=str), encoding="utf-8"
        )


def _update_task(task_id: str, **kwargs):
    if task_id in tasks:
        tasks[task_id].update(kwargs)
        _save_task(task_id)


def _public_task(t: dict[str, Any]) -> dict[str, Any]:
    dxnn_path = t.get("dxnn_path")
    return {
        "id": t.get("id"),
        "status": t.get("status"),
        "progress": t.get("progress", 0),
        "step": t.get("step", ""),
        "message": t.get("message", ""),
        "model_name": t.get("model_name"),
        "dxnn_name": t.get("dxnn_name") or (Path(dxnn_path).name if dxnn_path else None),
        "error": t.get("error"),
        "created_at": t.get("created_at"),
        "finished_at": t.get("finished_at"),
        "can_download": t.get("status") == "done" and bool(dxnn_path) and Path(dxnn_path).exists(),
    }


def _prune_work(task_dir: Path, keep: Path):
    """Drop bulky intermediates; keep only the converted .dxnn."""
    keep = keep.resolve()
    for p in task_dir.iterdir():
        if p.resolve() == keep:
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
        except Exception:
            pass


# ---- FastAPI ----
app = FastAPI(title="DXNN Converter")


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


@app.get("/api/status")
async def api_status():
    available = _dx_com_available()
    return {
        "dx_com_available": available,
        "dx_com_path": "python-package" if available else None,
        "tasks": len(tasks),
    }


@app.get("/api/history")
async def api_history():
    items = [_public_task(t) for t in tasks.values()]
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"items": items}


@app.post("/api/convert")
async def api_convert(model_file: UploadFile = File(...)):
    """Upload a .pt model and start conversion. Returns task_id."""
    if not model_file.filename.endswith(".pt"):
        return JSONResponse(
            status_code=400, content={"error": "只支持 .pt 文件"}
        )

    if not _dx_com_available():
        return JSONResponse(
            status_code=500,
            content={
                "error": "未检测到 dx_com Python 包。请重新构建镜像：docker compose up --build",
            },
        )

    task_id = uuid.uuid4().hex[:12]
    task_dir = _task_dir(task_id)

    model_path = task_dir / model_file.filename
    with open(model_path, "wb") as f:
        content = await model_file.read()
        f.write(content)

    tasks[task_id] = {
        "id": task_id,
        "status": "pending",
        "progress": 0,
        "step": "upload",
        "message": "已上传，等待转换",
        "model_name": model_file.filename,
        "input_size": 640,
        "dxnn_path": None,
        "dxnn_name": None,
        "error": None,
        "created_at": _now_iso(),
        "finished_at": None,
    }
    _save_task(task_id)

    import threading
    thread = threading.Thread(
        target=_run_conversion,
        args=(task_id, task_dir, model_path),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id}


def _run_conversion(task_id: str, task_dir: Path, model_path: Path):
    """Run the official Ultralytics DEEPX export pipeline.

    Uses ``yolo.export(format="deepx")`` which internally:
      - exports ONNX with correct opset
      - generates the proper DX-COM config (letterbox resize, real calibration)
      - compiles via dx_com Python package
      - saves metadata.yaml (class names, imgsz, task)

    Output: ``{stem}_deepx_model/`` directory containing .dxnn + config.json + metadata.yaml.
    We extract the .dxnn file for download.
    """
    input_size = 640
    try:
        model_stem = model_path.stem

        _update_task(task_id, status="running", step="onnx", progress=10,
                     message="正在使用官方 Ultralytics DEEPX 导出...")

        from ultralytics import YOLO

        model = YOLO(str(model_path))

        _update_task(task_id, progress=20,
                     message="调用 yolo.export(format='deepx')，包含 ONNX 导出 + DX-COM 编译...")

        export_path = model.export(
            format="deepx",
            imgsz=input_size,
            simplify=True,
        )
        export_path = Path(export_path)
        if not export_path.exists():
            raise RuntimeError(f"DEEPX 导出失败: {export_path}")

        _update_task(task_id, progress=80,
                     message=f"官方导出完成: {export_path}")

        # export_path is the {stem}_deepx_model/ directory
        dxnn_files = list(export_path.rglob("*.dxnn"))
        if not dxnn_files:
            raise RuntimeError(f"导出目录中未找到 .dxnn 文件: {export_path}")

        dxnn_path = dxnn_files[0]
        final_path = task_dir / f"{model_stem}.dxnn"
        shutil.copy2(str(dxnn_path), str(final_path))

        # Also keep metadata.yaml if present (class names etc.)
        meta_files = list(export_path.rglob("metadata.yaml")) + list(export_path.rglob("metadata.yml"))
        if meta_files:
            shutil.copy2(str(meta_files[0]), task_dir / f"{model_stem}_metadata.yaml")

        _prune_work(task_dir, final_path)

        _update_task(
            task_id, status="done", step="done", progress=100,
            message=f"转换完成 (官方导出): {final_path.name}",
            dxnn_path=str(final_path),
            dxnn_name=final_path.name,
            finished_at=_now_iso(),
        )

    except Exception as e:
        _update_task(
            task_id, status="failed", step="error", progress=0,
            message=str(e), error=str(e), finished_at=_now_iso(),
        )

@app.get("/api/task/{task_id}")
async def api_task_status(task_id: str):
    """Poll task status."""
    task = tasks.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "任务不存在"})
    return _public_task(task)


@app.get("/api/download/{task_id}")
async def api_download(task_id: str):
    """Download the converted .dxnn file."""
    task = tasks.get(task_id)
    if not task:
        return JSONResponse(status_code=404, content={"error": "任务不存在"})
    if task.get("status") != "done":
        return JSONResponse(status_code=400, content={"error": "转换未完成"})
    dxnn_path = Path(task["dxnn_path"]) if task.get("dxnn_path") else None
    if not dxnn_path or not dxnn_path.exists():
        return JSONResponse(status_code=404, content={"error": "文件不存在"})
    return FileResponse(
        str(dxnn_path),
        filename=dxnn_path.name,
        media_type="application/octet-stream",
    )


@app.on_event("startup")
async def _load_tasks():
    for f in TASKS_DIR.glob("*.json"):
        try:
            t = json.loads(f.read_text(encoding="utf-8"))
            if t.get("status") in ("pending", "running"):
                t["status"] = "failed"
                t["step"] = "error"
                t["error"] = "服务重启，转换中断"
                t["message"] = t["error"]
                t["finished_at"] = t.get("finished_at") or _now_iso()
                tasks[t["id"]] = t
                _save_task(t["id"])
            else:
                tasks[t["id"]] = t
        except Exception:
            pass


# ---- HTML page ----
INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DXNN 转换器</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, "Microsoft YaHei", sans-serif; background: #f5f5f5; color: #333; }
.container { max-width: 800px; margin: 40px auto; padding: 0 20px 48px; }
h1 { text-align: center; margin-bottom: 8px; font-size: 28px; }
.subtitle { text-align: center; color: #888; margin-bottom: 32px; font-size: 14px; }
.card { background: #fff; border-radius: 12px; padding: 32px; margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
h2 { font-size: 16px; margin-bottom: 16px; }
label { display: block; font-weight: 600; margin-bottom: 8px; font-size: 14px; }
input[type="file"] { width: 100%; padding: 10px; border: 2px dashed #ddd;
                     border-radius: 8px; margin-bottom: 16px; font-size: 14px; }
.hint { color: #999; font-size: 12px; margin-bottom: 16px; }
button { width: 100%; padding: 14px; background: #4f46e5; color: #fff;
         border: none; border-radius: 8px; font-size: 16px; font-weight: 600;
         cursor: pointer; transition: background 0.2s; }
button:hover { background: #4338ca; }
button:disabled { background: #aaa; cursor: not-allowed; }
.progress-head { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
.spinner { width: 18px; height: 18px; border: 3px solid #c7d2fe; border-top-color: #4f46e5;
           border-radius: 50%; animation: spin 0.8s linear infinite; flex-shrink: 0; }
@keyframes spin { to { transform: rotate(360deg); } }
.progress-bar { width: 100%; height: 10px; background: #eee; border-radius: 999px; overflow: hidden; }
.progress-fill { height: 100%; width: 0; background: #4f46e5; transition: width 0.3s; }
.progress-meta { display: flex; justify-content: space-between; margin-top: 8px;
                 font-size: 13px; color: #666; }
.progress-msg { margin-top: 12px; font-size: 14px; color: #4f46e5; font-weight: 600; }
.error-msg { color: #dc2626; background: #fef2f2; padding: 12px;
             border-radius: 8px; margin-top: 12px; font-size: 13px;
             white-space: pre-wrap; word-break: break-all; }
.warning { background: #fffbeb; border: 1px solid #fcd34d; color: #92400e;
           padding: 12px 16px; border-radius: 8px; margin-bottom: 24px; font-size: 13px; }
.empty { color: #999; font-size: 14px; text-align: center; padding: 24px 0; }
.history-item { display: flex; align-items: center; justify-content: space-between; gap: 16px;
                padding: 14px 0; border-bottom: 1px solid #f0f0f0; }
.history-item:last-child { border-bottom: none; padding-bottom: 0; }
.history-item.fresh { background: #ecfdf5; margin: 0 -16px; padding: 14px 16px; border-radius: 8px; }
.history-name { font-weight: 600; font-size: 14px; word-break: break-all; }
.history-sub { color: #888; font-size: 12px; margin-top: 4px; }
.badge { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 999px;
         margin-left: 8px; vertical-align: middle; }
.badge.done { background: #ecfdf5; color: #059669; }
.badge.failed { background: #fef2f2; color: #dc2626; }
.badge.running, .badge.pending { background: #eef2ff; color: #4f46e5; }
.dl { display: inline-block; padding: 8px 16px; background: #059669; color: #fff;
      text-decoration: none; border-radius: 8px; font-size: 13px; font-weight: 600;
      white-space: nowrap; }
.dl:hover { background: #047857; }
.dl.disabled { background: #ccc; pointer-events: none; }
</style>
</head>
<body>
<div class="container">
  <h1>DXNN 模型转换器</h1>
  <p class="subtitle">上传 YOLO .pt 模型 → 自动转换为 .dxnn → 下载</p>

  <div id="warning" class="warning" style="display:none;"></div>

  <div class="card">
    <label>选择 .pt 模型文件</label>
    <input type="file" id="modelFile" accept=".pt" />
    <div class="hint">YOLOv8 / YOLO11 / YOLOv5 等 ultralytics 支持的模型</div>
    <button id="convertBtn" onclick="startConvert()">开始转换</button>
  </div>

  <div id="statusBox" class="card" style="display:none;">
    <div class="progress-head">
      <div id="spinner" class="spinner"></div>
      <h2 id="progressTitle" style="margin:0;">正在转换</h2>
    </div>
    <div class="progress-bar"><div id="progressFill" class="progress-fill"></div></div>
    <div class="progress-meta">
      <span id="progressStep">准备中</span>
      <span id="progressPct">0%</span>
    </div>
    <div id="progressMsg" class="progress-msg"></div>
    <div id="errorMsg" class="error-msg" style="display:none;"></div>
  </div>

  <div class="card">
    <h2>转换历史</h2>
    <div id="historyList"><div class="empty">暂无转换记录</div></div>
  </div>
</div>

<script>
let currentTaskId = null;
let pollTimer = null;
let freshId = null;

const STEP_LABEL = {
  upload: '上传模型',
  onnx: '导出 ONNX',
  config: '生成配置',
  compile: '编译 DXNN',
  done: '完成',
  error: '失败',
};

function escapeHtml(s) {
  return String(s || '').replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[c]));
}

function fmtTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString('zh-CN', { hour12: false });
}

function statusLabel(s) {
  return { done: '已完成', failed: '失败', running: '转换中', pending: '等待中' }[s] || s;
}

async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (!data.dx_com_available) {
      const w = document.getElementById('warning');
      w.style.display = 'block';
      w.textContent = '未检测到 dx_com Python 包。请重新构建镜像：docker compose up --build';
      document.getElementById('convertBtn').disabled = true;
    }
  } catch(e) {}
}

function showProgress(task) {
  const box = document.getElementById('statusBox');
  box.style.display = 'block';
  const running = task.status === 'running' || task.status === 'pending';
  document.getElementById('spinner').style.display = running ? 'block' : 'none';
  document.getElementById('progressTitle').textContent = running ? '正在转换' : (task.status === 'done' ? '转换完成' : '转换失败');
  const pct = task.progress || 0;
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressFill').style.background = task.status === 'failed' ? '#dc2626' : '#4f46e5';
  document.getElementById('progressPct').textContent = pct + '%';
  document.getElementById('progressStep').textContent = STEP_LABEL[task.step] || task.step || '';
  document.getElementById('progressMsg').textContent = task.message || '';
  const em = document.getElementById('errorMsg');
  if (task.status === 'failed') {
    em.style.display = 'block';
    em.textContent = task.error || task.message || '未知错误';
  } else {
    em.style.display = 'none';
  }
}

function setBusy(busy, text) {
  const btn = document.getElementById('convertBtn');
  btn.disabled = busy;
  btn.textContent = text;
}

async function startConvert() {
  const modelFile = document.getElementById('modelFile').files[0];
  if (!modelFile) { alert('请选择 .pt 模型文件'); return; }

  setBusy(true, '上传中...');
  showProgress({ status: 'pending', step: 'upload', progress: 2, message: '正在上传 ' + modelFile.name });

  const formData = new FormData();
  formData.append('model_file', modelFile);

  try {
    const res = await fetch('/api/convert', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) {
      alert(data.error);
      setBusy(false, '开始转换');
      document.getElementById('statusBox').style.display = 'none';
      return;
    }
    currentTaskId = data.task_id;
    freshId = data.task_id;
    setBusy(true, '转换中...');
    pollStatus();
  } catch(e) {
    alert('上传失败: ' + e.message);
    setBusy(false, '开始转换');
    document.getElementById('statusBox').style.display = 'none';
  }
}

async function pollOnce() {
  if (!currentTaskId) return;
  const res = await fetch('/api/task/' + currentTaskId);
  const data = await res.json();
  showProgress(data);
  if (data.status === 'done' || data.status === 'failed') {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
    setBusy(false, '开始转换');
    await loadHistory();
    if (data.status === 'done') {
      setTimeout(() => { document.getElementById('statusBox').style.display = 'none'; }, 1200);
    }
  }
}

function pollStatus() {
  if (pollTimer) clearInterval(pollTimer);
  pollOnce();
  pollTimer = setInterval(pollOnce, 1500);
}

async function loadHistory() {
  try {
    const res = await fetch('/api/history');
    const data = await res.json();
    const list = document.getElementById('historyList');
    const items = data.items || [];
    if (!items.length) {
      list.innerHTML = '<div class="empty">暂无转换记录</div>';
      return;
    }
    list.innerHTML = items.map(item => {
      const cls = item.id === freshId ? 'history-item fresh' : 'history-item';
      const outName = item.dxnn_name || String(item.model_name || '').replace(/\\.pt$/i, '.dxnn');
      const err = item.status === 'failed' && item.error
        ? ' · ' + escapeHtml(String(item.error).split('\\n')[0]) : '';
      const dl = item.can_download
        ? `<a class="dl" href="/api/download/${item.id}">下载</a>`
        : `<span class="dl disabled">不可下载</span>`;
      return `<div class="${cls}">
        <div>
          <div class="history-name">${escapeHtml(item.model_name)} → ${escapeHtml(outName)}
            <span class="badge ${item.status}">${statusLabel(item.status)}</span>
          </div>
          <div class="history-sub">${fmtTime(item.created_at)}${err}</div>
        </div>
        ${dl}
      </div>`;
    }).join('');
  } catch(e) {
    document.getElementById('historyList').innerHTML = '<div class="empty">历史加载失败</div>';
  }
}

checkStatus();
loadHistory();
</script>
</body>
</html>
"""
