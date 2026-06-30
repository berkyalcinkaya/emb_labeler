"""Embryo ROI extraction via embpred_deploy's Faster-RCNN, with a center-crop fallback.

This backs the GUI's ``get_ROI``. The expensive detector lives in
:mod:`embpred_backend` (loaded once, shared with inference). Detected boxes are cached
per ``(depth, timepoint)`` in the patient's ``rcnn_boxes.json`` sidecar so scrolling
reuses them instead of re-running the detector.

Hot-path policy (priority #1 = labeling speed): :func:`roi_for_display` is synchronous
and NEVER runs the detector. It returns a cached RCNN crop when one exists, otherwise an
instant center crop. Actually running the detector happens off the UI thread via
:func:`detect_and_cache_frame` (single frame) or :func:`detect_boxes_for_patient`
(batch); the GUI refreshes the ROI view once a box is cached.

A ``null`` cache entry records a detection that fell back to a center crop, so we don't
re-run the detector on frames the model can't localise.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

CENTER_CROP_FRACTION = 0.5  # central half in each axis (matches the old placeholder)


def center_crop(image: np.ndarray, fraction: float = CENTER_CROP_FRACTION) -> np.ndarray:
    height, width = image.shape[:2]
    half = fraction / 2.0
    r0, r1 = int(height * (0.5 - half)), int(height * (0.5 + half))
    c0, c1 = int(width * (0.5 - half)), int(width * (0.5 + half))
    r0, r1 = max(0, r0), min(height, max(r0 + 1, r1))
    c0, c1 = max(0, c0), min(width, max(c0 + 1, c1))
    return image[r0:r1, c0:c1]


def crop_with_box(image: np.ndarray, box: Optional[List[int]]) -> np.ndarray:
    """Crop ``image`` to ``[x1, y1, x2, y2]``, clamped; center-crop on a bad/empty box."""
    if not box:
        return center_crop(image)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return center_crop(image)
    return image[y1:y2, x1:x2]


def roi_for_display(
    image: np.ndarray,
    patient_ts=None,
    depth_name: Optional[str] = None,
    timepoint: Optional[int] = None,
) -> Tuple[np.ndarray, bool]:
    """Return ``(roi_image, is_cached)`` without ever invoking the detector.

    ``is_cached`` is True when an RCNN box (or a recorded fallback) already exists for
    this frame; False means the caller may want to schedule background detection.
    """
    have_cache = (
        patient_ts is not None
        and depth_name is not None
        and timepoint is not None
        and patient_ts.has_rcnn_box(depth_name, timepoint)
    )
    if have_cache:
        box = patient_ts.get_rcnn_box(depth_name, timepoint)
        return crop_with_box(image, box), True
    return center_crop(image), False


def detect_and_cache_frame(
    image: np.ndarray,
    patient_ts=None,
    depth_name: Optional[str] = None,
    timepoint: Optional[int] = None,
    save: bool = True,
) -> Optional[List[int]]:
    """Run the detector on one frame and cache the result. Off-UI-thread only.

    Returns the detected box, or None when detection fell back to a center crop. The
    cache stores ``null`` for the fallback case so it isn't retried.
    """
    import embpred_backend as eb  # lazy: pulls in torch

    bbox = eb.detect_bbox(image)
    box = list(bbox) if bbox is not None else None
    if patient_ts is not None and depth_name is not None and timepoint is not None:
        patient_ts.set_rcnn_box(depth_name, timepoint, box, save=save)
    return box


def detect_boxes_for_patient(
    patient_ts,
    depths: Optional[List[str]] = None,
    overwrite: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> None:
    """Batch-detect ROI boxes for a patient and persist them. Off-UI-thread only.

    ``depths`` defaults to all of the patient's depths. Already-cached frames are
    skipped unless ``overwrite`` is True. Saves once at the end for efficiency.
    """
    import embpred_backend as eb  # lazy

    target_depths = depths if depths is not None else list(patient_ts.depths)
    num_timepoints = patient_ts.num_timepoints()

    work: List[Tuple[str, int]] = []
    for depth_name in target_depths:
        if depth_name not in patient_ts.depths:
            continue
        for timepoint in range(num_timepoints):
            if overwrite or not patient_ts.has_rcnn_box(depth_name, timepoint):
                work.append((depth_name, timepoint))

    total = len(work)
    for done, (depth_name, timepoint) in enumerate(work):
        if should_cancel is not None and should_cancel():
            break
        depth_index = patient_ts.depths.index(depth_name)
        image = patient_ts.get_image(timepoint, depth_index)
        bbox = eb.detect_bbox(image)
        patient_ts.set_rcnn_box(
            depth_name, timepoint, list(bbox) if bbox is not None else None, save=False
        )
        if progress_cb is not None:
            progress_cb(done + 1, total)

    patient_ts.save_rcnn_boxes()
