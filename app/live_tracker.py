from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .normalization import is_plausible_nigerian_plate, normalize_plate_text


@dataclass
class VehicleTrackState:
    track_id: int
    plate: str = ""
    ocr_confidence: float = 0.0
    detection_confidence: float = 0.0
    last_seen: float = 0.0
    last_ocr_time: float = 0.0
    recognition_history: list[dict[str, Any]] = field(default_factory=list)
    bbox: tuple[int, int, int, int] | None = None
    vehicle_make: str | None = None
    registration_recorded_plate: str = ""
    registration_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        confidence_percent = round(max(0.0, min(1.0, self.ocr_confidence)) * 100.0, 1)
        detection_percent = round(max(0.0, min(1.0, self.detection_confidence)) * 100.0, 1)
        filled = max(1, min(10, int(round(confidence_percent / 10.0))))
        bar = "█" * filled + "░" * (10 - filled)

        history = [
            {
                "plate": item["plate"],
                "ocr_confidence": round(float(item["confidence"]), 4),
                "ocr_confidence_percent": round(float(item["confidence"]) * 100.0, 1),
                "detection_confidence": round(float(item["detection_confidence"]), 4),
                "timestamp": item["timestamp"],
            }
            for item in self.recognition_history[-10:]
        ]

        return {
            "track_id": self.track_id,
            "plate": self.plate or "Reading...",
            "plate_status": "Tracked" if self.plate else "Reading...",
            "ocr_confidence": round(self.ocr_confidence, 4),
            "ocr_confidence_percent": confidence_percent,
            "detection_confidence": round(self.detection_confidence, 4),
            "detection_confidence_percent": detection_percent,
            "bbox": list(self.bbox) if self.bbox else None,
            "confidence_bar": bar,
            "last_seen": self.last_seen,
            "last_seen_seconds_ago": max(0.0, self.last_seen),
            "history_count": len(self.recognition_history),
            "recognition_history": history,
            "status": "Tracked" if self.plate else "Reading...",
            "vehicle_make": self.vehicle_make,
            "registration_status": "Registered" if self.registration_recorded_plate else "Reading...",
            "recognition_count": self.registration_count,
        }


class LiveVehicleTracker:
    def __init__(self, ocr_interval: int = 5, track_timeout: float = 10.0) -> None:
        self.tracked_vehicles: dict[int, VehicleTrackState] = {}
        self.next_track_id = 1
        self.ocr_interval = max(1, int(ocr_interval))
        self.track_timeout = max(1.0, float(track_timeout))

    def _next_id(self) -> int:
        track_id = self.next_track_id
        self.next_track_id += 1
        return track_id

    def observe(self, track_id: int | None, bbox: tuple[int, int, int, int] | list[int], detection_confidence: float, timestamp: float) -> VehicleTrackState:
        identifier = int(track_id) if track_id is not None else self._next_id()
        state = self.tracked_vehicles.get(identifier)
        if state is None:
            state = VehicleTrackState(track_id=identifier)
            self.tracked_vehicles[identifier] = state
        state.bbox = tuple(int(value) for value in bbox)
        state.detection_confidence = float(detection_confidence)
        state.last_seen = float(timestamp)
        return state

    def should_ocr(self, track_id: int, timestamp: float) -> bool:
        state = self.tracked_vehicles.get(track_id)
        if state is None:
            return False
        if not state.plate:
            return True
        return (timestamp - state.last_ocr_time) >= self.ocr_interval

    def record_ocr_result(self, track_id: int, plate_text: str, ocr_confidence: float, detection_confidence: float, timestamp: float | None = None) -> bool:
        normalized = normalize_plate_text(str(plate_text or ""))
        if not normalized or not is_plausible_nigerian_plate(normalized):
            return False

        state = self.tracked_vehicles.setdefault(track_id, VehicleTrackState(track_id=track_id))
        state.last_seen = float(timestamp if timestamp is not None else state.last_seen)
        result = {
            "plate": normalized,
            "confidence": float(ocr_confidence),
            "detection_confidence": float(detection_confidence),
            "timestamp": state.last_seen,
        }
        state.recognition_history.append(result)
        if len(state.recognition_history) > 20:
            state.recognition_history = state.recognition_history[-20:]

        if state.plate and normalized != state.plate and float(ocr_confidence) < max(state.ocr_confidence * 0.8, 0.35):
            return False

        state.plate, state.ocr_confidence = self._best_plate(state)
        state.last_ocr_time = state.last_seen
        return True

    def _best_plate(self, state: VehicleTrackState) -> tuple[str, float]:
        if not state.recognition_history:
            return "", 0.0

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in state.recognition_history:
            grouped[str(item["plate"])].append(item)

        best_plate = ""
        best_score = -1.0
        best_confidence = 0.0

        for plate, entries in grouped.items():
            confidence_sum = sum(float(item["confidence"]) for item in entries)
            max_confidence = max(float(item["confidence"]) for item in entries)
            score = confidence_sum + max_confidence + (0.25 * len(entries))
            if score > best_score:
                best_plate = plate
                best_score = score
                best_confidence = max_confidence

        return best_plate, best_confidence

    def prune(self, timestamp: float) -> None:
        expired = [track_id for track_id, state in self.tracked_vehicles.items() if (timestamp - state.last_seen) > self.track_timeout]
        for track_id in expired:
            del self.tracked_vehicles[track_id]

    def snapshot(self, timestamp: float | None = None) -> list[dict[str, Any]]:
        current_time = float(timestamp if timestamp is not None else 0.0)
        if timestamp is not None:
            self.prune(current_time)

        results = []
        for track_id in sorted(self.tracked_vehicles):
            state = self.tracked_vehicles[track_id]
            results.append(state.to_dict())
        return results
