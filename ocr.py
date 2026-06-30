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


def _usable_depths(patient_ts, reference_depths: Sequence[str]) -> List[str]:
    """The reference depths that actually exist (with images) for this patient."""
    return [
        d
        for d in reference_depths
        if d in patient_ts.depths and patient_ts.image_paths[patient_ts.depths.index(d)]
    ]


def _ocr_one_timepoint(patient_ts, timepoint: int, usable_depths: Sequence[str]) -> Tuple[Optional[float], Optional[str]]:
    """OCR a single timepoint, trying each depth in order. Returns ``(hours, depth)``."""
    for depth_name in usable_depths:
        depth_index = patient_ts.depths.index(depth_name)
        try:
            image = patient_ts.get_image(timepoint, depth_index)
        except (IndexError, OSError):
            continue
        value = extract_hours_from_image(image)
        if value is not None:
            return value, depth_name
    return None, None


def extract_hours_for_patient(
    patient_ts,
    reference_depths: Optional[Sequence[str]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """OCR every timepoint and return the ``ocr_times.json`` payload (no persistence).

    Tries each depth in ``reference_depths`` order per timepoint until one yields a
    number, then linearly interpolates the gaps. See :func:`compute_and_save_ocr_times`
    for the durable, resumable variant used by the GUI.
    """
    depths = list(reference_depths) if reference_depths else DEFAULT_REFERENCE_DEPTHS
    usable_depths = _usable_depths(patient_ts, depths)
    num_timepoints = patient_ts.num_timepoints()

    raw_hours: Dict[int, Optional[float]] = {}
    depth_used: Dict[str, str] = {}
    for timepoint in range(num_timepoints):
        if should_cancel is not None and should_cancel():
            break
        value, depth_name = _ocr_one_timepoint(patient_ts, timepoint, usable_depths)
        raw_hours[timepoint] = value
        if depth_name is not None:
            depth_used[str(timepoint)] = depth_name
        if progress_cb is not None:
            progress_cb(timepoint + 1, num_timepoints)

    return _build_ocr_payload(
        raw_hours, depth_used, num_timepoints, depths, patient_ts.image_fingerprint()
    )


# --------------------------------------------------------------------------------------
# Durable, resumable compute + persist.
#
# OCR is persisted to ``ocr_times.json`` every ``OCR_FLUSH_EVERY`` timepoints (and once
# at the end), and a restarted run *resumes* from whatever is already persisted for the
# same image set: timepoints already OCR'd (including ones that failed and were recorded
# as ``null``) are skipped. So toggling embryos mid-run — or the app closing — never
# discards progress.
# --------------------------------------------------------------------------------------
OCR_FLUSH_EVERY = 10


def _build_ocr_payload(
    raw_hours: Dict[int, Optional[float]],
    depth_used: Dict[str, str],
    num_timepoints: int,
    reference_depths: Sequence[str],
    fingerprint: Any,
) -> Dict[str, Any]:
    """Build the sidecar payload from the timepoints OCR'd so far.

    ``hours`` is interpolated across the full range (yet-to-be-processed timepoints read
    as gaps and refine as the run continues); ``raw_hours`` holds only processed
    timepoints so a resume can tell them apart from un-attempted ones.
    """
    raw_list = [raw_hours.get(i) for i in range(num_timepoints)]
    filled, interpolated = _interpolate(raw_list)
    return {
        "hours": {str(i): filled[i] for i in range(num_timepoints)},
        "raw_hours": {str(i): raw_hours[i] for i in sorted(raw_hours)},
        "interpolated": [str(i) for i in interpolated],
        "reference_depths": list(reference_depths),
        "depth_used": depth_used,
        "timestamp": _now_iso(),
        "image_fingerprint": fingerprint,
        "num_timepoints": num_timepoints,
        "processed_timepoints": len(raw_hours),
        "complete": len(raw_hours) >= num_timepoints and num_timepoints > 0,
    }


def _load_resumable_ocr(patient_ts, fingerprint: Any) -> Tuple[Dict[int, Optional[float]], Dict[str, str]]:
    """Per-timepoint OCR results already persisted for *this* image set.

    Returns ``({}, {})`` when nothing reusable exists (no file, or a different image
    set), forcing a fresh OCR pass.
    """
    data = patient_ts.load_ocr_times()
    if not data or data.get("image_fingerprint") != fingerprint:
        return {}, {}
    raw_hours: Dict[int, Optional[float]] = {
        int(key): value for key, value in (data.get("raw_hours") or {}).items()
    }
    depth_used: Dict[str, str] = dict(data.get("depth_used") or {})
    return raw_hours, depth_used


def compute_and_save_ocr_times(
    patient_ts,
    reference_depths: Optional[Sequence[str]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """Run OCR for the patient, persisting ``ocr_times.json`` incrementally.

    Continues from any results already persisted for the same image set when ``resume``
    is True (default); pass ``resume=False`` to force a fresh pass.
    """
    reference = list(reference_depths) if reference_depths else DEFAULT_REFERENCE_DEPTHS
    usable_depths = _usable_depths(patient_ts, reference)
    num_timepoints = patient_ts.num_timepoints()
    fingerprint = patient_ts.image_fingerprint()

    raw_hours, depth_used = _load_resumable_ocr(patient_ts, fingerprint) if resume else ({}, {})
    if progress_cb is not None:
        progress_cb(len(raw_hours), num_timepoints)

    def persist() -> Dict[str, Any]:
        return _build_ocr_payload(raw_hours, depth_used, num_timepoints, reference, fingerprint)

    if num_timepoints > 0 and len(raw_hours) >= num_timepoints:
        payload = persist()
        patient_ts.save_ocr_times(payload)
        return payload

    flushed_count = len(raw_hours)
    for timepoint in range(num_timepoints):
        if timepoint in raw_hours:
            continue
        if should_cancel is not None and should_cancel():
            break
        value, depth_name = _ocr_one_timepoint(patient_ts, timepoint, usable_depths)
        raw_hours[timepoint] = value
        if depth_name is not None:
            depth_used[str(timepoint)] = depth_name
        if len(raw_hours) - flushed_count >= OCR_FLUSH_EVERY:
            patient_ts.save_ocr_times(persist())
            flushed_count = len(raw_hours)
        if progress_cb is not None:
            progress_cb(len(raw_hours), num_timepoints)

    # Persist whatever was OCR'd, even if cancelled part-way; an immediate cancel with
    # nothing processed leaves the sidecar untouched.
    if not raw_hours:
        return patient_ts.load_ocr_times()
    payload = persist()
    patient_ts.save_ocr_times(payload)
    return payload


def has_ocr_times(patient_ts) -> bool:
    return bool(patient_ts.load_ocr_times())


def ocr_complete(patient_ts) -> bool:
    """True only when every timepoint has been OCR'd and persisted.

    A partial sidecar (an interrupted/resumable run) returns False so callers re-launch
    the worker to resume. Legacy sidecars without the ``complete`` flag are treated as
    complete when they carry hours.
    """
    data = patient_ts.load_ocr_times()
    if not data:
        return False
    if "complete" in data:
        return bool(data["complete"])
    return bool(data.get("hours"))


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
