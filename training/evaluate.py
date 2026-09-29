"""Evaluate trained YOLO license-plate weights on the configured test split."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/license_plate.pt"))
    parser.add_argument("--dataset", type=Path, default=Path("dataset/data.yaml"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Trained model not found: {args.model}. Run training first.")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before evaluation.") from exc
    metrics = YOLO(str(args.model)).val(data=str(args.dataset), split=args.split, imgsz=args.imgsz, device=args.device)
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
