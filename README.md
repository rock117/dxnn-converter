# DXNN 模型转换器

在 Windows 上用 Docker 跑一个网页界面，上传 YOLO `.pt` 模型，自动转换为 `.dxnn`，下载即可。

## 前提

1. **Docker Desktop** 已安装并运行
2. **DX-COM** — Dockerfile 会从 DEEPX SDK 索引安装 `dx_com` Python 包，无需单独下载编译器

## 启动

```bash
cd dxnn-converter
docker compose up --build
```

打开浏览器: **http://localhost:8899**

## 使用

详见 **[校准数据集与转换说明](docs/calibration-data.md)**。

1. 在 `.env` 的 `YOLO_HOST_PATH` 下放好校准数据集（含 `data.yaml` 与图片）
2. 选择 `.pt` 模型文件
3. 填写校准路径（相对 `/app/yolo`），例如 `data.yaml` — **必填，不上传数据集**
4. 点「开始转换」→ 完成后在「转换历史」下载 `.dxnn`

路径留空、不在映射目录内、或不存在时会直接报错。输入尺寸固定为 640。

## 转换流程

使用 **官方 Ultralytics DEEPX 导出**（`yolo.export(format="deepx")`）：

```
.pt → yolo.export(format="deepx")
         ├─ ONNX 导出 (自动 opset)
         ├─ DX-COM 编译 (letterbox 校准 + INT8 量化)
         └─ 输出 {stem}_deepx_model/ 含 .dxnn + config.json + metadata.yaml
```

官方导出会自动处理：
- letterbox resize（`mode: pad`, `pad_value: [114,114,114]`）— 与板端推理一致
- 真实校准数据集（默认 COCO128，可配 `data` 参数）
- metadata.yaml（类名、imgsz、task）

## 常见问题

### Q: 转换失败，提示 dx_com 相关错误

确保 Docker 构建时安装了 `dx_com` Python 包。Dockerfile 已配置从 DEEPX SDK 索引安装：

```bash
docker compose down
docker compose up --build
```

### Q: 转换失败，DX-COM 报错

查看网页上的错误信息。常见原因：
- 模型不是 ultralytics 格式（需要 `from ultralytics import YOLO` 能加载）
- 训练时输入尺寸不是 640
- DX-COM 仅支持 x86-64 Linux（不支持 ARM64 / Windows）

### Q: 转换完的 .dxnn 怎么用？

1. 上传到检测系统：`http://www.nnruibo.cn:18080/detect/` → 模型管理 → 上传模型
2. 用这个 `.dxnn` 模型做检测 → 自动走 DX-M1 NPU

## 目录结构

```
dxnn-converter/
├── app.py              # FastAPI 转换服务 + 网页
├── Dockerfile          # x86 Ubuntu + ultralytics[export-deepx] + DX-COM
├── docker-compose.yml  # 一键启动
├── requirements.txt    # Python 依赖
├── work/               # 运行时工作目录（自动创建）
└── tasks/              # 任务记录（自动创建）
```
