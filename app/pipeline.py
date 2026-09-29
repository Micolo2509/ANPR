from typing import Any
import logging
from pathlib import Path

import cv2

from .config import Settings, settings
from .detector import YOLOPlateDetector
from .normalization import compact_plate_text, is_ocr_quality_plate, normalize_plate_text, verify_plate_with_database
from .ocr import OCRReader
from .schemas import BoundingBox, PlateResult
from .segmentation.character_segmentation import segment_plate_characters
from .vehicle_classifier import VehicleClassifier


logger = logging.getLogger(__name__)


class VNPRPipeline:
    def __init__(
        self,
        detector: Any = None,
        ocr: Any = None,
        config: Settings = settings,
        vehicle_classifier: Any = None,
        plate_registry: Any = None,
    ) -> None:
        self.config = config
        self.detector = detector or YOLOPlateDetector(config.model_path, config.detection_confidence, config.detection_iou, config.detection_imgsz)
        self.ocr = ocr or OCRReader(config.ocr_languages, config.ocr_gpu, padding_pct=config.ocr_padding_pct, upscale_factor=config.ocr_upscale_factor)
        self.vehicle_classifier = vehicle_classifier or VehicleClassifier(config.vehicle_model_path)
        self.plate_registry = plate_registry
        self.last_detection_diagnostics: dict[str, Any] = {}

    @staticmethod
    def _pad_plate_crop(image: Any, detection: Any, padding_pct: float = 0.10) -> Any:
        if image is None or image.size == 0:
            return image

        height, width = image.shape[:2]
        crop_width = max(1, detection.x2 - detection.x1)
        crop_height = max(1, detection.y2 - detection.y1)
        pad_x = max(4, int(crop_width * padding_pct))
        pad_y = max(4, int(crop_height * padding_pct))

        x1 = max(0, detection.x1 - pad_x)
        y1 = max(0, detection.y1 - pad_y)
        x2 = min(width, detection.x2 + pad_x)
        y2 = min(height, detection.y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            return image[detection.y1:detection.y2, detection.x1:detection.x2]
        return image[y1:y2, x1:x2]

    def _run_character_segmentation(self, plate_crop: Any, debug: bool = False) -> dict[str, Any]:
        if plate_crop is None or plate_crop.size == 0:
            return {"enabled": True, "success": False, "character_count": 0, "boxes": [], "character_crops": [], "debug": {"reason": "empty_plate_crop"}}
        try:
            debug_dir = Path("debug_segmentation") if debug else None
            result = segment_plate_characters(plate_crop, debug=debug, debug_dir=debug_dir)
            if not isinstance(result, dict):
                return {"enabled": True, "success": False, "character_count": 0, "boxes": [], "character_crops": [], "debug": {"reason": "invalid_segmentation_result"}}
            result.setdefault("enabled", True)
            result.setdefault("success", False)
            result.setdefault("character_count", 0)
            result.setdefault("boxes", [])
            result.setdefault("character_crops", [])
            if not result.get("success"):
                result.setdefault("debug", {})
                result["debug"].setdefault("reason", "segmentation_failed")
            return result
        except Exception as exc:
            logger.warning("Character segmentation failed for plate crop: %s", exc, exc_info=True)
            return {"enabled": True, "success": False, "character_count": 0, "boxes": [], "character_crops": [], "debug": {"reason": str(exc)}}

    def recognize(self, image: Any, debug: bool = False) -> list[PlateResult]:
        results = []
        if hasattr(self.detector, "detect_robust") and self.config.detection_multiscale:
            try:
                detections = self.detector.detect_robust(
                    image,
                    imgsz_values=[640, 832, 1024, getattr(self.detector, "imgsz", 1280)],
                    tile_overlap=float(self.config.detection_tile_overlap),
                )
            except TypeError:
                detections = self.detector.detect_robust(image)
            self.last_detection_diagnostics = getattr(self.detector, "last_diagnostics", {})
        else:
            detections = self.detector.detect(image)
            self.last_detection_diagnostics = {
                "plate_detected": bool(detections),
                "detection_method": "yolo" if detections else None,
                "detections": [self.detector._serialize_detection(item) for item in detections],
            }
        vehicle = None
        for detection in detections:
            debug_payload = None
            image_shape = image.shape if image is not None else None
            color_format = "BGR" if image is not None and len(image.shape) == 3 else "GRAY"
            print(
                f"[OCR DEBUG] original image shape={image_shape}; color_format={color_format}; "
                f"bbox={ [detection.x1, detection.y1, detection.x2, detection.y2] }",
                flush=True,
            )
            crop = image[detection.y1:detection.y2, detection.x1:detection.x2]
            crop_is_none = crop is None
            crop_is_empty = crop is not None and crop.size == 0
            print(
                f"[OCR DEBUG] crop is None={crop_is_none}; crop is empty={crop_is_empty}; "
                f"crop shape={None if crop is None else crop.shape}; "
                f"crop width={None if crop is None else crop.shape[1]}; crop height={None if crop is None else crop.shape[0]}; "
                f"color_format={color_format}",
                flush=True,
            )
            segmentation_result: dict[str, Any] = {"enabled": True, "success": False, "character_count": 0, "boxes": [], "character_crops": [], "debug": {}}
            if crop is None or crop.size == 0:
                logger.warning("Plate crop is empty: width=0 height=0 bbox=%s", [detection.x1, detection.y1, detection.x2, detection.y2])
                text, ocr_confidence = "", 0.0
            else:
                crop_width, crop_height = crop.shape[1], crop.shape[0]
                logger.info("Plate crop: width=%s height=%s shape=%s bbox=%s", crop_width, crop_height, crop.shape, [detection.x1, detection.y1, detection.x2, detection.y2])
                debug_crop_path = Path("debug_plate_crop.jpg")
                saved = cv2.imwrite(str(debug_crop_path), crop)
                print(
                    f"[OCR DEBUG] saved crop to {debug_crop_path} -> {saved}; "
                    f"crop shape={crop.shape}; crop width={crop_width}; crop height={crop_height}",
                    flush=True,
                )
                segmentation_result = self._run_character_segmentation(crop, debug=debug)
                char_texts: list[str] = []
                if segmentation_result.get("success"):
                    try:
                        for i, char_crop in enumerate(segmentation_result.get("character_crops", [])[:12], start=1):
                            if char_crop is None or char_crop.size == 0:
                                continue
                            char_text, char_conf = self.ocr.read(char_crop)
                            cleaned = normalize_plate_text(char_text)
                            if cleaned and len(cleaned) <= 2:
                                char_texts.append(cleaned)
                        segmented_plate = "".join(char_texts)
                        if len(segmented_plate) >= 5 and is_ocr_quality_plate(segmented_plate):
                            text = segmented_plate
                            ocr_confidence = max(0.0, min(1.0, sum(float(item) for item in [0.9]) / 1.0))
                            segmentation_result["used_for_ocr"] = True
                            segmentation_result["segmented_text"] = segmented_plate
                        else:
                            text, ocr_confidence = self.ocr.read(crop)
                            segmentation_result["used_for_ocr"] = False
                    except Exception as exc:
                        logger.warning("Character-level OCR failed, falling back to full crop OCR: %s", exc, exc_info=True)
                        text, ocr_confidence = self.ocr.read(crop)
                        segmentation_result["used_for_ocr"] = False
                else:
                    text, ocr_confidence = self.ocr.read(crop)
                    segmentation_result["used_for_ocr"] = False
                ocr_debug = getattr(self.ocr, "last_debug", {})
                debug_payload = dict(ocr_debug) if isinstance(ocr_debug, dict) else {}
                debug_segmentation = dict(segmentation_result)
                debug_segmentation["character_crops"] = []
                debug_payload["character_segmentation"] = debug_segmentation
                if debug:
                    debug_payload["detection"] = self.last_detection_diagnostics
                normalized_ocr = normalize_plate_text(text)
                logger.info("OCR raw result=%r normalized result=%r confidence=%s", text, normalized_ocr, ocr_confidence)
                print(
                    f"[OCR DEBUG] Raw OCR result: {text!r}; Raw OCR confidence: {ocr_confidence}; "
                    f"Normalized OCR result: {normalized_ocr!r}",
                    flush=True,
                )

            ocr_debug = debug_payload if isinstance(debug_payload, dict) else {}
            raw_ocr_text = normalize_plate_text(ocr_debug.get("raw_ocr_text", text))
            normalized_text = normalize_plate_text(text)
            compact_text = compact_plate_text(normalized_text)
            ocr_succeeded = bool(compact_text and is_ocr_quality_plate(compact_text))
            ocr_uncertain = bool(compact_text and not ocr_succeeded) or bool(ocr_debug.get("raw_ocr_detections"))
            corrected_text = normalized_text
            correction_applied = bool(ocr_debug.get("correction_applied", raw_ocr_text != normalized_text))
            correction_reason = ocr_debug.get("correction_reason")

            if self.plate_registry and compact_text:
                corrected_text, correction_applied, correction_reason = verify_plate_with_database(normalized_text, self.plate_registry.lookup)
                if corrected_text and corrected_text != normalized_text:
                    normalized_text = corrected_text
                    ocr_succeeded = is_ocr_quality_plate(corrected_text)
                    ocr_uncertain = not ocr_succeeded
                elif corrected_text == normalized_text:
                    normalized_text = corrected_text

            if ocr_uncertain and correction_reason is None:
                correction_reason = "ocr_format_uncertain"
            if not compact_text:
                normalized_text = "UNKNOWN"
                corrected_text = "UNKNOWN"
                correction_applied = False
                correction_reason = "ocr_empty"
                ocr_confidence = 0.0

            if debug_payload is None:
                debug_payload = {}
            debug_payload.update({
                "raw_ocr_text": raw_ocr_text,
                "corrected_text": corrected_text,
                "correction_applied": correction_applied,
                "correction_reason": correction_reason,
                "ocr_status": "success" if ocr_succeeded else "uncertain" if ocr_uncertain else "failed",
            })

            if compact_text and vehicle is None:
                vehicle = self.vehicle_classifier.classify(image)

            is_registered, is_verified = (
                self.plate_registry.lookup(normalized_text)
                if self.plate_registry and normalized_text != "UNKNOWN" and ocr_succeeded
                else (False, False)
            )

            results.append(PlateResult(
                normalized_text,
                detection.confidence,
                ocr_confidence,
                BoundingBox(detection.x1, detection.y1, detection.x2, detection.y2),
                vehicle.make if vehicle else None,
                vehicle.model if vehicle else None,
                vehicle.confidence if vehicle else None,
                is_registered,
                is_verified,
                debug=debug_payload,
                detection_method=detection.method,
            ))
        logger.info("Pipeline: YOLO detections=%s OCR results=%s", len(detections), sum(result.text != "UNKNOWN" for result in results))
        print(
            f"[OCR DEBUG] Final OCR results: {sum(result.text != 'UNKNOWN' for result in results)}; "
            f"Pipeline results: {len(results)}",
            flush=True,
        )
        return results