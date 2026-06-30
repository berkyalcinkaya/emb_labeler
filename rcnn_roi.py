"""Embryo ROI extraction via embpred_deploy's Faster-RCNN, with a center-crop fallback.

This backs the GUI's ``get_ROI``. The expensive detector lives in
:mod:`embpred_backend` (loaded once, shared with inference). One box per timepoint is
detected at the :func:`reference_depth` and cached in the patient's ``rcnn_boxes.json``
sidecar, then reused for every focal depth. The *same* cache and detection
(:func:`detect_box_for_timepoint`) are shared with the timepoint-prediction pipeline, so
the embryo bbox is computed once and never duplicated across the two.

Hot-path policy (priority #1 = labeling speed): :func:`roi_for_display` is synchronous
and NEVER runs the detector. It returns a cached RCNN crop when one exists, otherwise an
instant center crop. Running the detector happens off the UI thread, per-frame on demand
or via :func:`detect_boxes_for_patient` (batch); the GUI refreshes the ROI view once a
box is cached.

A ``null`` cache entry records a detection that fell back to a center crop, so we don't
re-run the detector on frames the model can't localise.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

CENTER_CROP_FRACTION = 0.5  # central half in each axis (matches the old placeholder)

# ROI detection runs at a single *reference* focal depth (F0 when present) and the box is
# reused for every other depth: the embryo sits at the same position across focal planes,
# so one box per timepoint serves them all. This detects once per timepoint instead of
# once per depth. See :func:`reference_depth`.

# Persist the boxes cache every this-many newly-detected frames so a cancelled/closed run
# keeps its progress and a restart resumes from the first un-detected frame.
ROI_FLUSH_EVERY = 10


def reference_depth(patient_ts) -> Optional[str]:
    """The focal depth ROI boxes are detected at, and reused across all depths.

    F0 when present (the labeler's default view), else the first populated depth — i.e.
    the patient's :meth:`default_depth_index`. Returns None when the patient has no
    images at all.
    """
    if not patient_ts.populated_depth_indices():
        return None
    return patient_ts.get_depth_name(patient_ts.default_depth_index())


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


def _inference_triplet(patient_ts, timepoint: int) -> Optional[List[np.ndarray]]:
    """``[F-15, F0, F15]`` images for ``timepoint`` if all three depths exist, else None."""
    import embpred_backend as eb  # lazy: INFERENCE_DEPTHS

    images: List[np.ndarray] = []
    for depth_name in eb.INFERENCE_DEPTHS:
        if depth_name not in patient_ts.depths:
            return None
        depth_index = patient_ts.depths.index(depth_name)
        if not patient_ts.image_paths[depth_index]:
            return None
        images.append(patient_ts.get_image(timepoint, depth_index))
    return images


def detect_box_for_timepoint(patient_ts, timepoint: int, ref_index: int):
    """Detect a timepoint's embryo box the same way the prediction pipeline does.

    Uses the ``F-15/F0/F15`` triplet fallback (:func:`embpred_backend.detect_triplet_bbox`)
    when those depths exist so a box is identical no matter which task produced it; falls
    back to single-frame detection on the reference depth otherwise.
    """
    import embpred_backend as eb  # lazy: pulls in torch

    triplet = _inference_triplet(patient_ts, timepoint)
    if triplet is not None:
        return eb.detect_triplet_bbox(triplet)
    return eb.detect_bbox(patient_ts.get_image(timepoint, ref_index))


def detect_boxes_for_patient(
    patient_ts,
    overwrite: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    should_yield: Optional[Callable[[], bool]] = None,
) -> None:
    """Batch-detect ROI boxes at the reference depth and persist them. Off-UI-thread only.

    The box is reused for every focal depth (and shared with timepoint predictions via the
    same cache). Already-cached frames are skipped unless ``overwrite`` is True, so a
    restarted run resumes where it left off; progress is flushed every
    :data:`ROI_FLUSH_EVERY` detections (and at the end) so partial work survives
    cancellation / app close. ``should_yield`` lets a background run step aside between
    frames for a higher-priority (on-screen) task that shares the detector.
    """
    import embpred_backend as eb  # lazy: shared yield helper (+ the detector itself)

    ref_depth = reference_depth(patient_ts)
    if ref_depth is None:
        return
    ref_index = patient_ts.depths.index(ref_depth)
    num_timepoints = patient_ts.num_timepoints()

    work = [
        timepoint
        for timepoint in range(num_timepoints)
        if overwrite or not patient_ts.has_rcnn_box(ref_depth, timepoint)
    ]

    total = len(work)
    for done, timepoint in enumerate(work):
        if should_cancel is not None and should_cancel():
            break
        # Step aside (between frames) while a higher-priority task wants the detector.
        eb.wait_while_yield(should_yield, should_cancel)
        if should_cancel is not None and should_cancel():
            break
        bbox = detect_box_for_timepoint(patient_ts, timepoint, ref_index)
        patient_ts.set_rcnn_box(
            ref_depth, timepoint, list(bbox) if bbox is not None else None, save=False
        )
        if (done + 1) % ROI_FLUSH_EVERY == 0:
            patient_ts.save_rcnn_boxes()
        if progress_cb is not None:
            progress_cb(done + 1, total)

    patient_ts.save_rcnn_boxes()
