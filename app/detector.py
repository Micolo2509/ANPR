from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int = 0
    class_name: str = "license_plate"
    method: str = "yolo"


class YOLOPlateDetector:
    def __init__(self, model_path: Path | str, confidence: float = 0.25, iou: float = 0.45, imgsz: int = 1280) -> None:
        self.model_path = Path(model_path).resolve() if isinstance(model_path, str) else model_path.resolve()
        self.confidence = confidence
        self.iou = iou
        self.imgsz = imgsz
        self._model: Any = None
        self.last_diagnostics: dict[str, Any] = {}

    def _load(self) -> Any:
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Install ultralytics to use YOLO detection") from exc
            if not self.model_path.exists():
                raise FileNotFoundError(f"YOLO model not found: {self.model_path}")
            self._model = YOLO(str(self.model_path))
        return self._model

    def _class_name(self, class_id: int) -> str:
        names = getattr(self._load(), "names", {})
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _run_yolo(self, image: Any, confidence: float, imgsz: int, offset: tuple[int, int] = (0, 0)) -> list[Detection]:
        results = self._load()(image, conf=confidence, iou=self.iou, imgsz=imgsz, verbose=False)
        detections: list[Detection] = []
        offset_x, offset_y = offset
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            class_values = boxes.cls.cpu().tolist() if getattr(boxes, "cls", None) is not None else [0] * len(boxes.conf)
            for coordinates, confidence_value, class_value in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), class_values):
                class_id = int(class_value)
                x1, y1, x2, y2 = (max(0, int(value)) for value in coordinates)
                detections.append(Detection(x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y, float(confidence_value), class_id, self._class_name(class_id), "yolo"))
        return detections

    def detect(self, image: Any, confidence: float | None = None, imgsz: int | None = None) -> list[Detection]:
        conf = self.confidence if confidence is None else confidence
        size = self.imgsz if imgsz is None else imgsz
        return self._run_yolo(image, conf, size)

    @staticmethod
    def _iou(left: Detection, right: Detection) -> float:
        x1, y1 = max(left.x1, right.x1), max(left.y1, right.y1)
        x2, y2 = min(left.x2, right.x2), min(left.y2, right.y2)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if not intersection:
            return 0.0
        left_area = max(0, left.x2 - left.x1) * max(0, left.y2 - left.y1)
        right_area = max(0, right.x2 - right.x1) * max(0, right.y2 - right.y1)
        return intersection / max(1, left_area + right_area - intersection)

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        selected: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if all(self._iou(detection, existing) <= self.iou for existing in selected):
                selected.append(detection)
        return selected

    @staticmethod
    def _opencv_candidates(image: np.ndarray) -> list[Detection]:
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 180)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[Detection] = []
        image_area = float(width * height)
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            area = box_width * box_height
            aspect = box_width / max(1.0, float(box_height))
            if area / image_area < 0.0002 or area / image_area > 0.25 or not 1.8 <= aspect <= 8.5:
                continue
            edge_density = float(np.count_nonzero(edges[y:y + box_height, x:x + box_width])) / max(1, area)
            if edge_density < 0.04:
                continue
            score = min(0.49, 0.12 + min(0.20, edge_density) + min(0.17, area / image_area * 2.0))
            candidates.append(Detection(x, y, x + box_width, y + box_height, score, 0, "license_plate", "opencv_fallback"))
        return sorted(candidates, key=lambda item: item.confidence, reverse=True)[:10]

    def detect_robust(self, image: Any, confidence_values: list[float] | None = None, imgsz_values: list[int] | None = None, tile_overlap: float = 0.20) -> list[Detection]:
        height, width = image.shape[:2]
        confidences = list(dict.fromkeys(float(value) for value in (confidence_values or [self.confidence, 0.25, 0.20, 0.15, 0.10])))
        sizes = list(dict.fromkeys(int(value) for value in (imgsz_values or [640, 832, 1024, self.imgsz])))
        attempts: dict[str, int] = {}
        detections: list[Detection] = []
        yolo_available = True
        for size in sizes:
            for confidence in confidences:
                try:
                    current = self._run_yolo(image, confidence, size)
                except (FileNotFoundError, ImportError, RuntimeError):
                    yolo_available = False
                    current = []
                attempts[f"yolo_{size}_conf_{confidence:.2f}"] = len(current)
                if current:
                    detections = self._nms(current)
                    break
            if detections:
                break
        if not detections:
            tile_size = max(512, min(max(width, height), sizes[-1]))
            stride = max(1, int(tile_size * (1.0 - tile_overlap)))
            tiled: list[Detection] = []
            x_positions = sorted(set(list(range(0, max(1, width - tile_size + 1), stride)) + [max(0, width - tile_size)]))
            y_positions = sorted(set(list(range(0, max(1, height - tile_size + 1), stride)) + [max(0, height - tile_size)]))
            for y in y_positions:
                for x in x_positions:
                    x2, y2 = min(width, x + tile_size), min(height, y + tile_size)
                    try:
                        tiled.extend(self._run_yolo(image[y:y2, x:x2], confidences[-1], tile_size, (x, y)))
                    except (FileNotFoundError, ImportError, RuntimeError):
                        yolo_available = False
            attempts["tiled_yolo"] = len(tiled)
            detections = self._nms(tiled)
        else:
            attempts["tiled_yolo"] = 0
        fallback = self._opencv_candidates(image) if not detections else []
        attempts["opencv_fallback"] = len(fallback)
        if not detections:
            detections = fallback[:1]
        self.last_diagnostics = {
            "model_path": str(self.model_path),
            "image_size": {"width": width, "height": height},
            "class_names": getattr(self._load(), "names", {}) if yolo_available else {},
            "confidence_thresholds": confidences,
            "image_sizes": sizes,
            "detection_attempts": attempts,
            "detections": [self._serialize_detection(item) for item in detections],
            "plate_detected": bool(detections),
            "detection_method": detections[0].method if detections else None,
        }
        return detections

    @staticmethod
    def _serialize_detection(detection: Detection) -> dict[str, Any]:
        return {"class_id": detection.class_id, "class_name": detection.class_name, "confidence": detection.confidence, "bbox": [detection.x1, detection.y1, detection.x2, detection.y2], "width": detection.x2 - detection.x1, "height": detection.y2 - detection.y1, "detection_method": detection.method}

    def diagnose(self, image: Any, confidence_values: list[float] | None = None, imgsz_values: list[int] | None = None) -> dict[str, Any]:
        confidences = confidence_values or [0.35, 0.25, 0.20, 0.15, 0.10]
        sizes = imgsz_values or [640, 832, 1024, 1280]
        rows: list[dict[str, Any]] = []
        for conf in confidences:
            for size in sizes:
                detections = self.detect(image, confidence=conf, imgsz=size)
                rows.append({
                    "confidence": conf,
                    "imgsz": size,
                    "count": len(detections),
                    "detections": [
                        {
                            "class_id": d.class_id,
                            "class_name": d.class_name,
                            "bbox": [d.x1, d.y1, d.x2, d.y2],
                            "confidence": d.confidence,
                            "width": d.x2 - d.x1,
                            "height": d.y2 - d.y1,
                            "detection_method": d.method,
                        }
                        for d in detections
                    ],
                })
        return {
            "model_path": str(self.model_path),
            "default_confidence": self.confidence,
            "default_imgsz": self.imgsz,
            "class_names": getattr(self._load(), "names", {}),
            "image_size": {"width": int(image.shape[1]), "height": int(image.shape[0])},
            "results": rows,
        }