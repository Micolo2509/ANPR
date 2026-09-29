from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class CharacterSegmentationError(RuntimeError):
    """Raised when plate character segmentation is not possible."""


def _safe_crop(image: Any, x1: int, y1: int, x2: int, y2: int) -> Any:
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def _enhance_plate(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return image

    gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def _threshold_plate(gray: np.ndarray) -> np.ndarray:
    if gray is None or gray.size == 0:
        return gray

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.count_nonzero(otsu) > 0 and np.count_nonzero(otsu) < gray.size:
        return otsu

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10,
    )
    return thresh


def _morphology(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    return cleaned


def _filter_candidate_box(x: int, y: int, w: int, h: int, plate_width: int, plate_height: int) -> bool:
    if w <= 0 or h <= 0:
        return False
    if w < 4 or h < 8:
        return False
    if w > plate_width * 0.60 or h > plate_height * 0.95:
        return False
    if w < plate_width * 0.02 or h < plate_height * 0.12:
        return False
    aspect = w / max(1, h)
    if aspect < 0.15 or aspect > 1.2:
        return False
    return True


def _save_debug_image(path: str | Path, image: Any) -> None:
    if image is None or image.size == 0:
        return
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image)
    except Exception:
        pass


def segment_plate_characters(
    plate_crop: Any,
    debug: bool = False,
    debug_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Segment a detected plate crop into candidate character boxes.

    This function is intentionally conservative. It will fail gracefully and fall back to the
    existing end-to-end OCR path whenever the crop is empty, too small, or too ambiguous.
    """
    if plate_crop is None or plate_crop.size == 0:
        return {"enabled": True, "success": False, "character_count": 0, "boxes": [], "character_crops": [], "debug": {"reason": "empty_plate_crop"}}

    plate_rgb = plate_crop.copy()
    if len(plate_rgb.shape) == 2:
        plate_rgb = cv2.cvtColor(plate_rgb, cv2.COLOR_GRAY2BGR)

    height, width = plate_rgb.shape[:2]
    if height < 20 or width < 20:
        return {"enabled": True, "success": False, "character_count": 0, "boxes": [], "character_crops": [], "debug": {"reason": "plate_too_small"}}

    gray = _enhance_plate(plate_rgb)
    thresholded = _threshold_plate(gray)
    processed = _morphology(thresholded)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(processed, connectivity=8)
    debug_info: dict[str, Any] = {"original_shape": (int(width), int(height)), "reason": None}
    candidate_boxes: list[tuple[int, int, int, int]] = []

    for label_index in range(1, num_labels):
        x, y, box_w, box_h, area = stats[label_index]
        if area <= 12:
            continue
        if x <= 1 or y <= 1 or x + box_w >= width - 1 or y + box_h >= height - 1:
            continue
        if box_w < 3 or box_h < 8:
            continue
        if box_w > width * 0.55 or box_h > height * 0.8:
            continue
        aspect = box_w / max(1, box_h)
        if aspect < 0.15 or aspect > 1.2:
            continue
        if box_w < width * 0.02 or box_h < height * 0.10:
            continue
        candidate_boxes.append((x, y, box_w, box_h))

    if not candidate_boxes:
        debug_info["reason"] = "no_contours_found"
        if debug and debug_dir is not None:
            _save_debug_image(Path(debug_dir) / "plate_crop.jpg", plate_rgb)
            _save_debug_image(Path(debug_dir) / "grayscale_plate.jpg", gray)
            _save_debug_image(Path(debug_dir) / "thresholded_plate.jpg", thresholded)
        return {"enabled": True, "success": False, "character_count": 0, "boxes": [], "character_crops": [], "debug": debug_info}

    final_boxes = sorted(candidate_boxes, key=lambda item: item[0])
    character_crops: list[np.ndarray] = []
    boxes_out: list[dict[str, int]] = []

    for index, (x, y, box_w, box_h) in enumerate(final_boxes):
        pad_x = max(2, int(box_w * 0.20))
        pad_y = max(2, int(box_h * 0.25))
        crop = _safe_crop(plate_rgb, x - pad_x, y - pad_y, x + box_w + pad_x, y + box_h + pad_y)
        if crop is None or crop.size == 0:
            continue
        character_crops.append(crop)
        boxes_out.append({
            "x": int(x),
            "y": int(y),
            "width": int(box_w),
            "height": int(box_h),
            "index": int(index),
        })

    success = len(character_crops) >= 3
    debug_info["character_count"] = len(character_crops)
    debug_info["boxes"] = boxes_out

    if debug and debug_dir is not None:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        _save_debug_image(debug_dir / "plate_crop.jpg", plate_rgb)
        _save_debug_image(debug_dir / "grayscale_plate.jpg", gray)
        _save_debug_image(debug_dir / "thresholded_plate.jpg", thresholded)
        contour_debug = plate_rgb.copy()
        for x, y, box_w, box_h in final_boxes:
            cv2.rectangle(contour_debug, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
        _save_debug_image(debug_dir / "segmented_plate.jpg", contour_debug)
        for index, crop in enumerate(character_crops, start=1):
            _save_debug_image(debug_dir / f"character_{index}.jpg", crop)

    if not success:
        debug_info["reason"] = "too_few_characters_detected"

    return {"enabled": True, "success": success, "character_count": len(character_crops), "boxes": boxes_out, "character_crops": character_crops, "debug": debug_info}
