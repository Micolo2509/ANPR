from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class PlateResult:
    text: str
    detection_confidence: float
    ocr_confidence: float
    bbox: BoundingBox
    make: str | None = None
    model: str | None = None
    vehicle_confidence: float | None = None
    is_registered: bool = False
    is_verified: bool = False
    debug: dict[str, Any] | None = None
    detection_method: str = "yolo"

    @property
    def registration_status(self) -> str:
        if self.is_registered and self.is_verified:
            return "registered_verified"
        if self.is_registered:
            return "registered_unverified"
        return "not_registered_unverified"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["registration_status"] = self.registration_status
        result["plate_number"] = self.text
        result["plate_text"] = self.text
        result["ocr_status"] = self.debug.get("ocr_status", "success") if self.debug else ("success" if self.text != "UNKNOWN" else "failed")
        return result