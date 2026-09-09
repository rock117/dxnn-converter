# 校准数据集与 DXNN 转换说明

本文说明如何用**同分布校准数据**把 YOLO `.pt` 转成 DEEPX `.dxnn`，避免自定义目标（如某种水果）转换后漏检。

## 为什么必须填校准路径

`.pt` → `.dxnn` 会做 **INT8 量化**。量化需要真实图片估计激活范围。

- 校准图应与训练/部署场景**同分布**（同类目标、相近场景）。
- **不上传**数据集：把数据放到 `.env` 里 `YOLO_HOST_PATH` 对应目录，网页只填路径。
- **路径必填**。留空会直接报错，不会回退到默认 `coco8`。
- 路径必须在映射目录内，且文件必须存在；否则直接报错。

## 目录映射

| 位置 | 路径 |
|------|------|
| 宿主机 | `.env` 中的 `YOLO_HOST_PATH` |
| 容器内 | `/app/yolo` |

示例（`.env`）：

```env
YOLO_HOST_PATH=C:/rock/code/ai-test/yolo-detect-electroscope
```

把校准数据集放在该目录下，例如：

```
YOLO_HOST_PATH/
  data.yaml
  dataset/
    images/
      train/
      val/          # 校准主要用 val（或 yaml 里指定的 split）
    labels/
      ...
```

`data.yaml` 示例：

```yaml
path: dataset
train: images/train
val: images/val

names:
  0: class_a
  1: class_b
```

`path` 相对 `data.yaml` 所在目录解析，一般无需写成绝对路径。

## 网页怎么填

1. `docker compose up -d`（确保已配置 `YOLO_HOST_PATH`）
2. 打开 http://localhost:8899
3. 选择 `.pt` 模型
4. **校准数据集路径**（必填）填写相对 `/app/yolo` 的路径

| 网页填写 | 容器实际路径 |
|----------|----------------|
| `data.yaml` | `/app/yolo/data.yaml` |
| `fruit/data.yaml` | `/app/yolo/fruit/data.yaml` |
| `/app/yolo/data.yaml` | 也可，会自动去掉 `/app/yolo/` 前缀 |

5. 点「开始转换」→ 完成后下载 `.dxnn`

### 报错规则

| 情况 | 行为 |
|------|------|
| 路径留空 | 直接报错，要求填写 |
| 路径不在 `/app/yolo`（映射目录）内 | 直接报错 |
| 文件/目录不存在 | 直接报错 |
| 不是 `.yaml` / `.yml` 文件 | 直接报错 |

网页在输入/失焦时会调用 `GET /api/check-data-path?path=...` 提前检查：

1. `data.yaml` 文件本身是否存在（且位于映射目录内）
2. yaml 内 `path` / `train` / `val` 指向的目录是否存在
3. `val` 目录下是否至少有 1 张图片

点「开始转换」前也会再校验一次。导出时会把相对路径展开成绝对路径再交给 Ultralytics，避免工作目录不同导致找不到数据。

## 命令行方式（可选）

与网页等价，在容器内执行：

```powershell
docker compose exec converter python -c "
from ultralytics import YOLO
model = YOLO('/app/work/your.pt')  # 或 /app/yolo/xxx.pt
print(model.export(format='deepx', imgsz=640, simplify=True, data='/app/yolo/data.yaml'))
"
```

或：

```powershell
docker compose exec converter yolo export `
  model=/app/yolo/best.pt `
  format=deepx `
  imgsz=640 `
  simplify=True `
  data=/app/yolo/data.yaml
```

## 校准图建议

- 与部署场景同域（同物体、相近光照/背景）
- 建议 **≥100～300** 张；太少容易量化后漏检
- 导出后可用同一 `data.yaml` 对比：

```powershell
docker compose exec converter yolo val model=/app/yolo/best.pt data=/app/yolo/data.yaml
docker compose exec converter yolo val model=/app/yolo/best_deepx_model data=/app/yolo/data.yaml
```

## 相关配置

- `docker-compose.yml`：`${YOLO_HOST_PATH}:/app/yolo`
- `.env` / `.env.example`：`YOLO_HOST_PATH=...`
- `app.py`：表单字段 `data_path` → `model.export(..., data=...)`
