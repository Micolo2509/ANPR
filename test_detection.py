"""Run the plate detector diagnostics on one image and save debug_detection.jpg."""

import argparse
from pathlib import Path

import cv2

from app.config import settings
from app.detector import YOLOPlateDetector


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Nigerian plate detection")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("debug_detection.jpg"))
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Unable to read image: {args.image}")

    detector = YOLOPlateDetector(settings.model_path, settings.detection_confidence, settings.detection_iou, settings.detection_imgsz)
    diagnostic = detector.diagnose(image, [0.35, 0.25, 0.20, 0.15, 0.10], [640, 832, 1024, 1280])
    detections = detector.detect_robust(image, [0.35, 0.25, 0.20, 0.15, 0.10], [640, 832, 1024, 1280])

    debug_image = image.copy()
    for detection in detections:
        color = (0, 255, 0) if detection.method == "yolo" else (0, 165, 255)
        cv2.rectangle(debug_image, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 3)
        label = f"PLATE {detection.confidence:.2f} {detection.method}"
        cv2.putText(debug_image, label, (detection.x1, max(24, detection.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if not cv2.imwrite(str(args.output), debug_image):
        raise SystemExit(f"Unable to write debug image: {args.output}")

    print(f"Image size: {image.shape[1]} x {image.shape[0]}")
    print(f"Model: {diagnostic['model_path']}")
    print(f"Class names: {diagnostic['class_names']}")
    print(f"Confidence thresholds: {diagnostic['confidence_thresholds']}")
    print(f"Image sizes: {diagnostic['image_sizes']}")
    print("Detection attempts:")
    for key, count in detector.last_diagnostics.get("detection_attempts", {}).items():
        print(f"  {key} -> {count}")
    print(f"Best detections: {detector.last_diagnostics.get('detections', [])}")
    print(f"Detection method: {detector.last_diagnostics.get('detection_method')}")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())