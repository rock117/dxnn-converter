"""DXNN Converter — FastAPI web service.

Upload a .pt YOLO model, convert it to .dxnn for DEEPX DX-M1 NPU.
Pipeline: .pt → yolo.export(format="deepx", data=...) → .dxnn

Calibration dataset is NOT uploaded. Put it under YOLO_HOST_PATH (.env),
which is mounted at /app/yolo, then pass a path relative to that mount
(e.g. data.yaml or my_fruit/data.yaml).
"""
import importlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# ---- Paths ----
BASE_DIR = Path(__file__).parent
WORK_DIR = BASE_DIR / "work"
WORK_DIR.mkdir(exist_ok=True)
# Docker: ${YOLO_HOST_PATH}:/app/yolo — host datasets live here
YOLO_DIR = Path("/app/yolo")


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
        "data_path": t.get("data_path"),
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


def _resolve_data_path(raw: str | None) -> Path:
    """Resolve calibration dataset path under /app/yolo (Docker mount).

    Accepts paths relative to YOLO_DIR (preferred), or absolute paths that
    stay inside YOLO_DIR. Empty input raises ValueError (path is required).
    """
    if raw is None or not str(raw).strip():
        raise ValueError(
            "请填写校准数据集路径（相对 /app/yolo，例如 data.yaml），不可留空。"
        )
    text = str(raw).strip().replace("\\", "/")

    yolo = YOLO_DIR.resolve()
    p = Path(text)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        # strip leading ./ or /app/yolo/ if user pastes container path prefix
        for prefix in ("/app/yolo/", "app/yolo/"):
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break
        resolved = (yolo / text).resolve()

    try:
        resolved.relative_to(yolo)
    except ValueError as e:
        raise ValueError(
            f"校准数据路径必须位于映射目录内（容器 /app/yolo，对应 .env 的 YOLO_HOST_PATH）: {raw}"
        ) from e

    if not resolved.exists():
        raise ValueError(
            f"校准数据不存在: {resolved}（请把数据集放到 YOLO_HOST_PATH 下，"
            f"网页填写相对路径，例如 data.yaml）"
        )
    if resolved.is_file() and resolved.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"校准数据应为 .yaml/.yml 数据集配置文件: {resolved.name}")
    return resolved


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _load_yaml_file(path: Path) -> dict[str, Any]:
    try:
        from ultralytics.utils import YAML

        data = YAML.load(path)
    except Exception:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"无效的 YAML 内容: {path.name}")
    return data


def _dataset_root_from_yaml(yaml_path: Path, data: dict[str, Any]) -> Path:
    """Resolve dataset root: prefer path relative to the yaml file directory."""
    raw = data.get("path")
    if not raw:
        return yaml_path.parent.resolve()
    p = Path(str(raw))
    if p.is_absolute():
        return p.resolve()
    # User datasets almost always mean relative to the yaml location
    return (yaml_path.parent / p).resolve()


def _count_images(dir_path: Path) -> int:
    if not dir_path.is_dir():
        return 0
    n = 0
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMG_EXTS:
            n += 1
    return n


def _split_path(root: Path, value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    out: list[Path] = []
    for item in items:
        p = Path(str(item))
        out.append(p.resolve() if p.is_absolute() else (root / p).resolve())
    return out


def _validate_data_yaml(yaml_path: Path) -> dict[str, Any]:
    """Check data.yaml exists and configured path/train/val directories are present."""
    data = _load_yaml_file(yaml_path)
    if "val" not in data and "validation" in data:
        data["val"] = data["validation"]

    missing_keys = [k for k in ("train", "val") if k not in data or data.get(k) in (None, "")]
    if missing_keys:
        raise ValueError(
            f"{yaml_path.name} 缺少必要字段: {', '.join(missing_keys)}（需要 train 与 val）"
        )
    if data.get("names") is None and "nc" not in data:
        raise ValueError(f"{yaml_path.name} 缺少 names 或 nc")

    root = _dataset_root_from_yaml(yaml_path, data)
    if not root.exists():
        raise ValueError(
            f"{yaml_path.name} 中 path 不存在: {data.get('path')!r} → {root}"
        )
    if not root.is_dir():
        raise ValueError(f"{yaml_path.name} 中 path 不是目录: {root}")

    splits: dict[str, Any] = {}
    errors: list[str] = []
    for key in ("train", "val", "test"):
        if key not in data or data.get(key) in (None, ""):
            continue
        paths = _split_path(root, data[key])
        for sp in paths:
            exists = sp.exists()
            images = _count_images(sp) if exists and sp.is_dir() else 0
            if exists and sp.is_file() and sp.suffix.lower() in _IMG_EXTS:
                images = 1
            entry = {
                "configured": str(data[key]),
                "resolved": str(sp),
                "exists": exists,
                "images": images,
            }
            splits.setdefault(key, [])
            if isinstance(splits[key], list):
                splits[key].append(entry)
            if not exists:
                errors.append(f"{key} 路径不存在: {data[key]!r} → {sp}")
            elif sp.is_dir() and images < 1 and key == "val":
                errors.append(f"val 目录下没有图片: {sp}")

    # Flatten single-path splits for simpler API/UI
    flat_splits = {
        k: (v[0] if isinstance(v, list) and len(v) == 1 else v) for k, v in splits.items()
    }

    if errors:
        raise ValueError("；".join(errors))

    val_info = flat_splits.get("val")
    val_images = 0
    if isinstance(val_info, dict):
        val_images = int(val_info.get("images") or 0)
    elif isinstance(val_info, list):
        val_images = sum(int(x.get("images") or 0) for x in val_info)

    return {
        "root": str(root),
        "root_configured": data.get("path"),
        "splits": flat_splits,
        "val_images": val_images,
        "names": data.get("names"),
        "nc": data.get("nc") if data.get("nc") is not None else (
            len(data["names"]) if isinstance(data.get("names"), (list, dict)) else None
        ),
    }


def _materialize_data_yaml(yaml_path: Path, dest: Path) -> Path:
    """Write a copy of data.yaml with absolute path/train/val for reliable export."""
    data = _load_yaml_file(yaml_path)
    if "val" not in data and "validation" in data:
        data["val"] = data.pop("validation")
    root = _dataset_root_from_yaml(yaml_path, data)
    data["path"] = str(root)
    for key in ("train", "val", "test", "minival"):
        if key not in data or data.get(key) in (None, ""):
            continue
        paths = _split_path(root, data[key])
        data[key] = str(paths[0]) if len(paths) == 1 else [str(p) for p in paths]

    # Prefer ultralytics YAML saver when available; fallback to PyYAML
    try:
        from ultralytics.utils import YAML

        YAML.save(dest, data)
    except Exception:
        import yaml  # type: ignore

        dest.write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    return dest


# ---- FastAPI ----
app = FastAPI(title="DXNN Converter")


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


@app.get("/api/status")
async def api_status():
    available = _dx_com_available()
    yolo_ok = YOLO_DIR.is_dir()
    return {
        "dx_com_available": available,
        "dx_com_path": "python-package" if available else None,
        "yolo_dir": str(YOLO_DIR),
        "yolo_mounted": yolo_ok,
        "tasks": len(tasks),
    }


@app.get("/api/history")
async def api_history():
    items = [_public_task(t) for t in tasks.values()]
    items.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    return {"items": items}


@app.get("/api/check-data-path")
async def api_check_data_path(path: str = ""):
    """Validate calibration dataset yaml and the path/train/val it references."""
    raw = (path or "").strip()
    if not raw:
        return {
            "ok": False,
            "exists": False,
            "path": "",
            "resolved": None,
            "details": None,
            "error": "请填写校准数据集路径（相对 /app/yolo，例如 data.yaml）",
        }
    if not YOLO_DIR.is_dir():
        return {
            "ok": False,
            "exists": False,
            "path": raw,
            "resolved": None,
            "details": None,
            "error": "未挂载 /app/yolo，请检查 .env 的 YOLO_HOST_PATH",
        }
    try:
        resolved = _resolve_data_path(raw)
        details = _validate_data_yaml(resolved)
        try:
            display = str(resolved.relative_to(YOLO_DIR.resolve()))
        except ValueError:
            display = str(resolved)
        return {
            "ok": True,
            "exists": True,
            "path": raw,
            "resolved": display,
            "details": details,
            "error": None,
        }
    except ValueError as e:
        return {
            "ok": False,
            "exists": False,
            "path": raw,
            "resolved": None,
            "details": None,
            "error": str(e),
        }


@app.post("/api/convert")
async def api_convert(
    model_file: UploadFile = File(...),
    data_path: str = Form(""),
):
    """Upload a .pt model and start conversion. Returns task_id.

    data_path: required path under /app/yolo (YOLO_HOST_PATH), e.g. ``data.yaml``.
    Not uploaded — dataset must already exist on the mounted host folder.
    Empty path is rejected (no default coco8).
    """
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

    if not str(data_path or "").strip():
        return JSONResponse(
            status_code=400,
            content={
                "error": "请填写校准数据集路径（相对 /app/yolo，例如 data.yaml）。"
                "数据集需放在 .env 的 YOLO_HOST_PATH 映射目录下，不可留空。",
            },
        )

    try:
        resolved_data = _resolve_data_path(data_path)
        _validate_data_yaml(resolved_data)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    task_id = uuid.uuid4().hex[:12]
    task_dir = _task_dir(task_id)

    model_path = task_dir / model_file.filename
    with open(model_path, "wb") as f:
        content = await model_file.read()
        f.write(content)

    try:
        data_display = str(resolved_data.relative_to(YOLO_DIR.resolve()))
    except ValueError:
        data_display = str(resolved_data)

    tasks[task_id] = {
        "id": task_id,
        "status": "pending",
        "progress": 0,
        "step": "upload",
        "message": "已上传，等待转换",
        "model_name": model_file.filename,
        "data_path": data_display,
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
        args=(task_id, task_dir, model_path, resolved_data),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id, "data_path": data_display}


def _run_conversion(
    task_id: str,
    task_dir: Path,
    model_path: Path,
    data_path: Path | None = None,
):
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

        if data_path is None:
            raise ValueError("校准数据集路径不能为空")

        details = _validate_data_yaml(data_path)
        export_data = _materialize_data_yaml(data_path, task_dir / "calib_data.yaml")

        export_kwargs: dict[str, Any] = {
            "format": "deepx",
            "imgsz": input_size,
            "simplify": True,
            "data": str(export_data),
        }
        _update_task(
            task_id,
            progress=20,
            message=(
                f"调用 yolo.export(format='deepx', data={data_path}) "
                f"[val 图片 {details.get('val_images', '?')} 张] ..."
            ),
        )

        export_path = model.export(**export_kwargs)
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
input[type="text"] { width: 100%; padding: 10px 12px; border: 1px solid #ddd;
                     border-radius: 8px; margin-bottom: 8px; font-size: 14px; }
.hint { color: #999; font-size: 12px; margin-bottom: 16px; }
.path-status { font-size: 12px; margin: -8px 0 16px; min-height: 1.2em; }
.path-status.ok { color: #059669; }
.path-status.bad { color: #dc2626; }
.path-status.checking { color: #888; }
input[type="text"].path-bad { border-color: #fca5a5; }
input[type="text"].path-ok { border-color: #6ee7b7; }
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

    <label>校准数据集路径（相对 /app/yolo，必填）</label>
    <input type="text" id="dataPath" placeholder="例如 data.yaml 或 fruit/data.yaml" required />
    <div id="dataPathStatus" class="path-status" aria-live="polite"></div>
    <div class="hint" id="dataHint">
      不上传数据集。把数据放到 .env 的 YOLO_HOST_PATH 目录下，这里填相对路径。
      不可留空；路径必须在映射目录内且文件存在，否则直接报错。
    </div>

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
let dataPathOk = false;
let dataPathCheckTimer = null;
let dataPathCheckSeq = 0;

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

function setDataPathStatus(kind, text) {
  const el = document.getElementById('dataPathStatus');
  const input = document.getElementById('dataPath');
  el.className = 'path-status' + (kind ? ' ' + kind : '');
  el.textContent = text || '';
  input.classList.toggle('path-ok', kind === 'ok');
  input.classList.toggle('path-bad', kind === 'bad');
}

async function checkDataPath(forceEmptyMsg) {
  const raw = (document.getElementById('dataPath').value || '').trim();
  const seq = ++dataPathCheckSeq;
  if (!raw) {
    dataPathOk = false;
    if (forceEmptyMsg) {
      setDataPathStatus('bad', '请填写校准数据集路径');
    } else {
      setDataPathStatus('', '');
    }
    return false;
  }
  setDataPathStatus('checking', '正在检查路径…');
  try {
    const res = await fetch('/api/check-data-path?path=' + encodeURIComponent(raw));
    const data = await res.json();
    if (seq !== dataPathCheckSeq) return dataPathOk;
    if (data.ok) {
      dataPathOk = true;
      const d = data.details || {};
      const val = d.splits && d.splits.val;
      let extra = '';
      if (val && !Array.isArray(val)) {
        extra = ` · val: ${val.images} 张图`;
      } else if (typeof d.val_images === 'number') {
        extra = ` · val: ${d.val_images} 张图`;
      }
      if (d.root_configured != null) {
        extra += ` · path: ${d.root_configured}`;
      }
      setDataPathStatus('ok', '配置有效：/app/yolo/' + data.resolved + extra);
      return true;
    }
    dataPathOk = false;
    setDataPathStatus('bad', data.error || '路径无效');
    return false;
  } catch (e) {
    if (seq !== dataPathCheckSeq) return dataPathOk;
    dataPathOk = false;
    setDataPathStatus('bad', '路径检查失败: ' + e.message);
    return false;
  }
}

function scheduleCheckDataPath() {
  if (dataPathCheckTimer) clearTimeout(dataPathCheckTimer);
  dataPathCheckTimer = setTimeout(() => checkDataPath(false), 350);
}

async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    const w = document.getElementById('warning');
    const hints = [];
    if (!data.dx_com_available) {
      hints.push('未检测到 dx_com Python 包。请重新构建镜像：docker compose up --build');
      document.getElementById('convertBtn').disabled = true;
    }
    if (!data.yolo_mounted) {
      hints.push('未挂载 /app/yolo。请在 .env 配置 YOLO_HOST_PATH 后 docker compose up -d');
    } else {
      document.getElementById('dataHint').textContent =
        '数据集放在宿主机 YOLO_HOST_PATH 下，对应容器 ' + data.yolo_dir +
        '。填写相对路径，例如 data.yaml。输入后会自动检查是否存在。';
    }
    if (hints.length) {
      w.style.display = 'block';
      w.textContent = hints.join(' ');
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
  const dataPath = (document.getElementById('dataPath').value || '').trim();
  if (!dataPath) {
    setDataPathStatus('bad', '请填写校准数据集路径');
    alert('请填写校准数据集路径（相对 /app/yolo，例如 data.yaml）');
    return;
  }
  const ok = await checkDataPath(true);
  if (!ok) {
    alert(document.getElementById('dataPathStatus').textContent || '校准数据集路径无效');
    return;
  }

  setBusy(true, '上传中...');
  showProgress({ status: 'pending', step: 'upload', progress: 2, message: '正在上传 ' + modelFile.name });

  const formData = new FormData();
  formData.append('model_file', modelFile);
  formData.append('data_path', dataPath);

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
      const calib = item.data_path
        ? ' · 校准: ' + escapeHtml(item.data_path)
        : ' · 校准: 默认';
      const dl = item.can_download
        ? `<a class="dl" href="/api/download/${item.id}">下载</a>`
        : `<span class="dl disabled">不可下载</span>`;
      return `<div class="${cls}">
        <div>
          <div class="history-name">${escapeHtml(item.model_name)} → ${escapeHtml(outName)}
            <span class="badge ${item.status}">${statusLabel(item.status)}</span>
          </div>
          <div class="history-sub">${fmtTime(item.created_at)}${calib}${err}</div>
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

const dataPathInput = document.getElementById('dataPath');
dataPathInput.addEventListener('input', scheduleCheckDataPath);
dataPathInput.addEventListener('blur', () => checkDataPath(true));
dataPathInput.addEventListener('change', () => checkDataPath(true));
</script>
</body>
</html>
"""
