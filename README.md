# yolov11friutsclassify

GitHub 侧用于存放**水果检测**推理权重（**YOLO11n**，Ultralytics），与同一流水线 **v11** 一路一致。

**移动端对接**：见 **[docs/MOBILE_APP_INTEGRATION.md](docs/MOBILE_APP_INTEGRATION.md)**。  
可选 Secrets：`FRUIT_SERVER_UPLOAD_URL`、`FRUIT_SERVER_UPLOAD_TOKEN`（与其它水果检测仓库一致）。

## 当前权重

| 文件 | 说明 |
|------|------|
| `models/yolo11n_fruit_detect_best.pt` | 检测 checkpoint（**YOLO11n**） |

- **任务**：`detect`  
- **类别数**：6  
- **类别名**：见 `models/classes.json`  
- **来源 run**：`fruit_detection_formal100_20260430_164036`（与 v3 / v5 / v8 同批 formal100）  
- **训练起点**（`args.yaml`）：`yolo11n.pt`  
- **本地原路径**：`YOLO/versions/v11/runs/train/fruit_detection_formal100_20260430_164036/weights/best.pt`

## 本地推理

```bash
pip install ultralytics
python scripts/infer_fruit_detect.py incoming/your.jpg
```

## GitHub Actions

向 **`incoming/`** 推送图片触发 **Fruit detection (YOLO11n)**，或 **Run workflow**。

## 克隆

```bash
git clone git@github.com:yhlkxkzs/yolov11friutsclassify.git
```
