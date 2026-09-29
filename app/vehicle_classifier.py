from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VehiclePrediction:
    make: str
    model: str | None
    confidence: float


class VehicleClassifier:
    """Optional YOLO classifier for vehicle make/model labels.

    Classification labels should use ``make|model`` (for example,
    ``Toyota|Corolla``). A label without ``|`` is returned as the make.
    """

    def __init__(self, model_path: Path | None) -> None:
        self.model_path = model_path
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            if self.model_path is None:
                return None
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Install ultralytics to use vehicle classification") from exc
            if not self.model_path.exists():
                return None
            self._model = YOLO(str(self.model_path))
        return self._model

    def classify(self, image: Any) -> VehiclePrediction | None:
        model = self._load()
        if model is None:
            return None
        results = model(image, verbose=False)
        if not results:
            return None
        probabilities = getattr(results[0], "probs", None)
        names = getattr(results[0], "names", {})
        if probabilities is not None:
            top_index = int(probabilities.top1)
            confidence = float(probabilities.top1conf)
            label = str(names.get(top_index, top_index)).strip()
        else:
            boxes = getattr(results[0], "boxes", None)
            if boxes is None:
                return None
            predictions = []
            for class_id, confidence in zip(boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
                label = str(names.get(int(class_id), int(class_id))).strip()
                if label.lower() != "plate_number":
                    predictions.append((label, float(confidence)))
            if not predictions:
                return None
            label, confidence = max(predictions, key=lambda prediction: prediction[1])
        make, separator, model_name = label.partition("|")
        return VehiclePrediction(
            make=make.strip(),
            model=model_name.strip() if separator and model_name.strip() else None,
            confidence=confidence,
        )