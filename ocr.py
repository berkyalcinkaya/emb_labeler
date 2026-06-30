"""OCR of the embedded time-lapse clock (hours) from the bottom-right of each frame.

Ported from ``Notebook_OCR.ipynb``: crop a small bottom-right corner (~44 px, which
works for both 500x500 and 800x800 frames), run RapidOCR, and parse the hour value
with the regex ``(\\d+(?:\\.\\d+)?)\\s*h``. Reads the reference depth F0 by default and
retries other depths when a frame fails. Frames that never yield a number are filled by
linear interpolation from neighbours and flagged.

This is a pure backend module (no Qt). RapidOCR is imported lazily and the engine is
cached, since it is heavy to construct. Intended to be run once per patient on a worker
thread with a progress callback.
"""

from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Size of the bottom-right corner crop, in pixels (notebook used ~44 px).
CORNER_PX = 45

# Default order of depths to try for each timepoint: F0 first, then nearest depths.
DEFAULT_REFERENCE_DEPTHS = ["F0", "F-15", "F15", "F-30", "F30", "F-45", "F45"]

_HOUR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*h", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

_engine = None
_engine_lock = threading.Lock()


class OCRUnavailableError(RuntimeError):
    """Raised when RapidOCR cannot be imported."""


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise OCRUnavailableError(
                    "RapidOCR is required for OCR time extraction. Install it with "
                    "`pip install rapidocr onnxruntime`."
                ) from exc
            _engine = RapidOCR()
    return _engine


def _to_uint8_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float64)
        max_val = float(arr.max()) if arr.size else 0.0
        if max_val > 255.0:
            arr = arr / max_val * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def corner_crop(image: np.ndarray, corner_px: int = CORNER_PX) -> np.ndarray:
    """Bottom-right corner crop where the timestamp is rendered."""
    gray = _to_uint8_gray(image)
    height, width = gray.shape[:2]
    r0 = max(0, height - corner_px)
    c0 = max(0, width - corner_px)
    # Drop the last row/col (1 px border), mirroring the notebook's [..:H-1] slices.
    return gray[r0 : max(r0 + 1, height - 1), c0 : max(c0 + 1, width - 1)]


def parse_hours_from_text(text: str) -> Optional[float]:
    match = _HOUR_RE.search(text)
    if match:
        return float(match.group(1))
    match = _NUMBER_RE.search(text)
    return float(match.group()) if match else None


def extract_hours_from_image(image: np.ndarray) -> Optional[float]:
    """Run OCR on a frame's timestamp corner and parse the hour value (or None)."""
    engine = _get_engine()
    crop = corner_crop(image)
    result = engine(crop)
    texts = getattr(result, "txts", None)
    if not result or not texts:
        return None
    return parse_hours_from_text(" ".join(texts))


def _interpolate(values: List[Optional[float]]) -> Tuple[List[Optional[float]], List[int]]:
    """Linear-interpolate missing entries from known neighbours.

    Returns ``(filled_values, interpolated_indices)``. Entries with no known value
    anywhere remain None. Leading/trailing gaps are filled by edge clamping (np.interp).
    """
    known_x = [i for i, v in enumerate(values) if v is not None]
    if not known_x:
        return list(values), []
    known_y = [float(values[i]) for i in known_x]
    filled: List[Optional[float]] = list(values)
    interpolated: List[int] = []
    for i, value in enumerate(values):
        if value is None:
            filled[i] = float(np.interp(i, known_x, known_y))
            interpolated.append(i)
    return filled, interpolated


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def extract_hours_for_patient(
    patient_ts,
    reference_depths: Optional[Sequence[str]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """OCR every timepoint and return the ``ocr_times.json`` payload.

    Tries each depth in ``reference_depths`` order per timepoint until one yields a
    number, then linearly interpolates the gaps. Does not persist — see
    :func:`compute_and_save_ocr_times`.
    """
    depths = list(reference_depths) if reference_depths else DEFAULT_REFERENCE_DEPTHS
    usable_depths = [d for d in depths if d in patient_ts.depths and patient_ts.image_paths[patient_ts.depths.index(d)]]
    num_timepoints = patient_ts.num_timepoints()

    raw_hours: List[Optional[float]] = [None] * num_timepoints
    depth_used: Dict[str, str] = {}

    for timepoint in range(num_timepoints):
        if should_cancel is not None and should_cancel():
            break
        for depth_name in usable_depths:
            depth_index = patient_ts.depths.index(depth_name)
            try:
                image = patient_ts.get_image(timepoint, depth_index)
            except (IndexError, OSError):
                continue
            value = extract_hours_from_image(image)
            if value is not None:
                raw_hours[timepoint] = value
                depth_used[str(timepoint)] = depth_name
                break
        if progress_cb is not None:
            progress_cb(timepoint + 1, num_timepoints)

    filled, interpolated = _interpolate(raw_hours)

    return {
        "hours": {str(i): filled[i] for i in range(num_timepoints)},
        "raw_hours": {str(i): raw_hours[i] for i in range(num_timepoints)},
        "interpolated": [str(i) for i in interpolated],
        "reference_depths": depths,
        "depth_used": depth_used,
        "timestamp": _now_iso(),
        "image_fingerprint": patient_ts.image_fingerprint(),
    }


def compute_and_save_ocr_times(patient_ts, should_cancel=None, **kwargs) -> Dict[str, Any]:
    """Run OCR for the patient and persist ``ocr_times.json``.

    Returns ``{}`` without saving if cancelled mid-run.
    """
    ocr_times = extract_hours_for_patient(patient_ts, should_cancel=should_cancel, **kwargs)
    if should_cancel is not None and should_cancel():
        return {}
    patient_ts.save_ocr_times(ocr_times)
    return ocr_times


def has_ocr_times(patient_ts) -> bool:
    return bool(patient_ts.load_ocr_times())


def ocr_stale(patient_ts) -> bool:
    """True if OCR times exist but were computed against a different image set."""
    data = patient_ts.load_ocr_times()
    if not data:
        return False
    stored = data.get("image_fingerprint")
    if stored is None:
        return False
    return stored != patient_ts.image_fingerprint()


def load_hours_array(patient_ts) -> Optional[Dict[str, Any]]:
    """Return ``{"hours": [...], "interpolated": set(...), "num_timepoints": N}`` or None.

    ``hours`` is a list indexed by timepoint (values may be None where OCR failed and
    no interpolation was possible).
    """
    data = patient_ts.load_ocr_times()
    hours_map = data.get("hours") if isinstance(data, dict) else None
    if not hours_map:
        return None
    indices = sorted(int(key) for key in hours_map)
    num_timepoints = (max(indices) + 1) if indices else 0
    hours: List[Optional[float]] = [None] * num_timepoints
    for key, value in hours_map.items():
        hours[int(key)] = value
    interpolated = {int(key) for key in data.get("interpolated", [])}
    return {"hours": hours, "interpolated": interpolated, "num_timepoints": num_timepoints}
