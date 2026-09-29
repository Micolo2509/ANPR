"""Train a custom YOLO license-plate detector after validating the dataset."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

try:
    from .check_dataset import validate_dataset
except ImportError:
    from check_dataset import validate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="runs/detect/runs/detect/nigerian_license_plate-4/weights/last.pt", help="Base YOLO model...")
    parser.add_argument("--device", default="cpu", help="cpu, 0, 1, or another Ultralytics device value")
    parser.add_argument("--project", type=Path, default=Path("runs/detect"))
    parser.add_argument("--name", default="nigerian_license_plate")
    args = parser.parse_args()

    errors = validate_dataset(args.dataset)
    if errors:
        print("Dataset check failed; training was not started:")
        print("\n".join(f"- {error}" for error in errors))
        return 1

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before training.") from exc

    model = YOLO(args.model)
    results = model.train(
        data=str((args.dataset / "data.yaml").resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        resume=True,
    )
    best_weights = Path(model.trainer.best.pt)
    if not best_weights.is_file():
        raise SystemExit(f"Training finished without expected weights: {best_weights}")
    output_path = Path("models/license_plate.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, output_path)
    print(f"Saved trained model to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
