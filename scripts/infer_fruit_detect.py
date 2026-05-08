#!/usr/bin/env python3
"""对单张或多张图片做 YOLO11n 目标检测推理（Ultralytics，与训练 run 一致）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "yolo11n_fruit_detect_best.pt"
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".ppm", ".tif", ".tiff"}


def collect_images(incoming: Path) -> list[Path]:
    if not incoming.exists():
        return []
    out: list[Path] = []
    for p in sorted(incoming.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT:
            out.append(p)
    return out


def predict_one(model: YOLO, image_path: Path, imgsz: int, conf: float) -> dict:
    results = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, verbose=False)
    r = results[0]
    dets: list[dict] = []
    if r.boxes is not None and len(r.boxes):
        names = r.names
        for b in r.boxes:
            cid = int(b.cls.item())
            dets.append(
                {
                    "class_id": cid,
                    "class": names[cid] if isinstance(names, dict) else names[cid],
                    "confidence": float(b.conf.item()),
                    "xyxy": [float(x) for x in b.xyxy[0].tolist()],
                }
            )
    return {
        "image": str(image_path.as_posix()),
        "detections": dets,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="YOLO11n 水果检测推理（formal100 run）")
    p.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Ultralytics 检测权重 .pt",
    )
    p.add_argument("--imgsz", type=int, default=640, help="推理尺寸（与训练 imgsz 一致）")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument(
        "--incoming",
        type=Path,
        default=REPO_ROOT / "incoming",
        help="待检测图片目录（递归扫描）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "output" / "predictions.json",
        help="预测结果 JSON 路径",
    )
    p.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="可选：直接指定若干图片路径；留空则扫描 --incoming",
    )
    args = p.parse_args()

    weights = args.weights if args.weights.is_absolute() else REPO_ROOT / args.weights
    if not weights.exists():
        print(f"权重不存在: {weights}", file=sys.stderr)
        return 1

    if args.images:
        paths = []
        for x in args.images:
            xp = x if x.is_absolute() else REPO_ROOT / x
            if not xp.exists():
                print(f"图片不存在: {xp}", file=sys.stderr)
                return 1
            paths.append(xp)
    else:
        paths = collect_images(args.incoming)

    if not paths:
        print("未找到图片：请将图片放入 incoming/ 或通过参数传入路径。", file=sys.stderr)
        return 0

    model = YOLO(str(weights))
    results = [predict_one(model, im, args.imgsz, args.conf) for im in paths]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "weights": str(weights),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "count": len(results),
        "predictions": results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
