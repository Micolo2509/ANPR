"""Validate a one-class YOLO detection dataset before training."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ID = 0


def image_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def validate_split(dataset_root: Path, split: str) -> list[str]:
    errors: list[str] = []
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    if not images_dir.is_dir():
        return [f"Missing image directory: {images_dir}"]
    if not labels_dir.is_dir():
        return [f"Missing label directory: {labels_dir}"]

    images = image_files(images_dir)
    if not images:
        errors.append(f"No images found in {images_dir}")
    image_stems = {image.stem for image in images}
    label_files = {label.stem: label for label in labels_dir.glob("*.txt")}
    for image in images:
        label_path = labels_dir / f"{image.stem}.txt"
        if not label_path.exists():
            errors.append(f"Missing label for image: {image}")
        image_data = cv2.imread(str(image))
        if image_data is None:
            errors.append(f"Unreadable image: {image}")

    for stem, label_path in label_files.items():
        if stem not in image_stems:
            errors.append(f"Label has no matching image: {label_path}")
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                errors.append(f"{label_path}:{line_number}: expected 5 fields")
                continue
            try:
                class_id = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError:
                errors.append(f"{label_path}:{line_number}: fields must be numeric")
                continue
            if class_id != CLASS_ID:
                errors.append(f"{label_path}:{line_number}: class must be 0")
            if not all(0.0 <= value <= 1.0 for value in coordinates):
                errors.append(f"{label_path}:{line_number}: coordinates must be between 0 and 1")
            if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
                errors.append(f"{label_path}:{line_number}: width and height must be positive")
    return errors


def validate_dataset(dataset_root: Path) -> list[str]:
    errors: list[str] = []
    for split in ("train", "val", "test"):
        errors.extend(validate_split(dataset_root, split))
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.is_file():
        errors.append(f"Missing dataset config: {data_yaml}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    args = parser.parse_args()
    errors = validate_dataset(args.dataset)
    if errors:
        print("Dataset check failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print(f"Dataset check passed: {args.dataset.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
