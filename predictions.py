"""Model-prediction pipeline: run embpred inference over a patient, postprocess into
the strict stage sequence, map deploy classes onto labeler classes, and build the
``predictions.json`` / ``prediction_metadata.json`` sidecar payloads.

All heavy lifting (torch, RCNN, the model itself) lives in :mod:`embpred_backend`;
this module is the orchestration + schema layer. Persistence goes through
:class:`data.PatientTimeSeries` so there are no file paths or Qt widgets here.

Class mapping (Phase 0 deliverable)
-----------------------------------
embpred_deploy predicts these 14 classes (index order = developmental progression,
which ``monotonic_decoding`` relies on)::

    0 t1   1 tPN  2 tPNf  3 t2  4 t3  5 t4  6 t5
    7 t6   8 t7   9 t8   10 tM 11 tB 12 tEB 13 tEmpty

The labeler's human-label ``CLASSES`` use the *same names* (after fixing the old
``"tEB" "tB"`` typo), so the deploy->labeler mapping is identity-by-name. The mapping
is still expressed explicitly below so any future divergence is one obvious edit and
``DEPLOY_TO_LABELER_OVERRIDES`` is the single place to record a rename/merge.

Depth mapping
-------------
Inference uses the 3-depth subset ``F-15, F0, F15`` (``embpred_backend.INFERENCE_DEPTHS``)
while the labeler keeps all 7 depths for labeling / focal-depth navigation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import embpred_backend as eb
import rcnn_roi  # reference_depth + shared ROI-box cache (one-directional import; no cycle)

# Explicit deploy-name -> labeler-name overrides. Empty today because the names match
# 1:1; add an entry here if embpred ever renames/merges a class relative to the labeler.
DEPLOY_TO_LABELER_OVERRIDES: Dict[str, str] = {}

POSTPROCESS_METHOD = "monotonic_decoding/NLL"


def map_deploy_to_labeler(deploy_name: str) -> str:
    return DEPLOY_TO_LABELER_OVERRIDES.get(deploy_name, deploy_name)


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - np.max(scores)
    exp = np.exp(scores)
    total = exp.sum()
    return exp / total if total > 0 else np.full_like(exp, 1.0 / exp.size)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------
# Building triplet paths for the 3-depth subset.
# --------------------------------------------------------------------------------------
def _inference_triplet_paths(patient_ts) -> List[tuple]:
    """``[(F-15_path, F0_path, F15_path), ...]`` aligned to labeler timepoint indices."""
    depth_to_paths = {}
    for depth_name in eb.INFERENCE_DEPTHS:
        if depth_name not in patient_ts.depths:
            raise ValueError(f"Patient has no '{depth_name}' depth required for inference.")
        index = patient_ts.depths.index(depth_name)
        paths = patient_ts.image_paths[index]
        if not paths:
            raise ValueError(f"Patient depth folder '{depth_name}' has no images.")
        depth_to_paths[depth_name] = paths

    num_timepoints = patient_ts.num_timepoints()
    triplets = []
    for timepoint in range(num_timepoints):
        triplets.append(tuple(depth_to_paths[depth][timepoint] for depth in eb.INFERENCE_DEPTHS))
    return triplets


# --------------------------------------------------------------------------------------
# Compute + persist.
#
# Predictions are written to the sidecars *incrementally* (every ``PRED_FLUSH_EVERY``
# timepoints and once more at the end), and computation *resumes* from whatever is
# already persisted for the same image set + model. So a run interrupted by the user
# toggling embryos — or the app closing — is never lost: restarting the worker reads the
# sidecar and continues from the first un-scored timepoint instead of from scratch.
# --------------------------------------------------------------------------------------
PRED_FLUSH_EVERY = 10


def _load_resumable_raw_scores(patient_ts, model_name: str, fingerprint: Any) -> Dict[int, List[float]]:
    """Per-timepoint raw scores already persisted for *this* image set + model.

    Returns ``{}`` when nothing reusable exists (no predictions yet, a different image
    set, or a different model), forcing a fresh compute.
    """
    meta = patient_ts.load_prediction_metadata()
    if meta.get("image_fingerprint") != fingerprint or meta.get("model_name") != model_name:
        return {}
    predictions = patient_ts.load_predictions()
    timepoints = predictions.get("timepoints") if isinstance(predictions, dict) else None
    if not isinstance(timepoints, dict):
        return {}
    resumable: Dict[int, List[float]] = {}
    for key, entry in timepoints.items():
        scores = entry.get("raw_scores") if isinstance(entry, dict) else None
        if scores is not None:
            resumable[int(key)] = [float(v) for v in scores]
    return resumable


def _persist_predictions(
    patient_ts,
    raw_by_tp: Dict[int, List[float]],
    num_timepoints: int,
    class_names: List[str],
    labeler_names: List[str],
    model_name: str,
    fingerprint: Any,
) -> Dict[str, Any]:
    """Write predictions + metadata from the scores gathered so far (durable snapshot).

    Postprocessing (monotonic decoding) is global, so it is recomputed over the
    contiguous prefix of timepoints scored to date; intermediate snapshots refine as
    more timepoints arrive and converge to the final sequence once complete.
    """
    indices = sorted(raw_by_tp)
    raw_matrix = (
        np.array([raw_by_tp[i] for i in indices], dtype=np.float64)
        if indices
        else np.empty((0, eb.NCLASS), dtype=np.float64)
    )
    post_indices = eb.postprocess_monotonic(raw_matrix) if raw_matrix.shape[0] else np.empty((0,), dtype=int)

    timepoints: Dict[str, Any] = {}
    for position, timepoint in enumerate(indices):
        scores = np.asarray(raw_by_tp[timepoint], dtype=np.float64)
        probs = _softmax(scores)
        raw_index = int(np.argmax(scores))
        post_index = int(post_indices[position])
        timepoints[str(timepoint)] = {
            "raw_scores": [float(v) for v in scores],
            "raw_class_index": raw_index,
            "raw_class": labeler_names[raw_index],
            "post_class_index": post_index,
            "post_class": labeler_names[post_index],
            "max_prob": float(probs.max()),
        }

    predictions = {
        "class_names": labeler_names,
        "deploy_class_names": class_names,
        "timepoints": timepoints,
    }
    metadata = {
        "model_name": model_name,
        "rcnn_weights": eb.RCNN_FILENAME,
        "embpred_version": eb.embpred_version(),
        "timestamp": _now_iso(),
        "depth_subset": list(eb.INFERENCE_DEPTHS),
        "postprocess": POSTPROCESS_METHOD,
        "num_timepoints": int(num_timepoints),
        "computed_timepoints": len(indices),
        "complete": len(indices) >= num_timepoints and num_timepoints > 0,
        "image_fingerprint": fingerprint,
    }
    patient_ts.save_predictions(predictions)
    patient_ts.save_prediction_metadata(metadata)
    return predictions


def compute_and_save_predictions(
    patient_ts,
    model_name: str,
    use_gpu: Optional[bool] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    should_yield: Optional[Callable[[], bool]] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    """Run inference for ``patient_ts``, persisting predictions incrementally.

    Writes ``predictions.json`` / ``prediction_metadata.json`` through the patient's
    durable sidecar I/O, flushing partial progress as it goes. Never touches
    ``labels.json``. When ``resume`` is True (default) it continues from any scores
    already persisted for the same image set + model; pass ``resume=False`` to force a
    full recompute (e.g. a deliberate "re-run"). ``should_yield`` lets a background run
    step aside between timepoints for a higher-priority (on-screen) task. Returns the
    predictions dict.
    """
    triplets = _inference_triplet_paths(patient_ts)
    num_timepoints = len(triplets)
    class_names = [eb.DEPLOY_CLASS_MAPPING[i] for i in range(eb.NCLASS)]
    labeler_names = [map_deploy_to_labeler(name) for name in class_names]
    fingerprint = patient_ts.image_fingerprint()

    raw_by_tp: Dict[int, List[float]] = (
        _load_resumable_raw_scores(patient_ts, model_name, fingerprint) if resume else {}
    )
    if progress_cb is not None:
        progress_cb(len(raw_by_tp), num_timepoints)

    # Everything already scored for this image set + model: just normalise the snapshot.
    if num_timepoints > 0 and len(raw_by_tp) >= num_timepoints:
        return _persist_predictions(
            patient_ts, raw_by_tp, num_timepoints, class_names, labeler_names, model_name, fingerprint
        )

    # The RCNN box each timepoint needs is the same one the ROI display caches, so reuse
    # it: read the shared cache, detect once on a miss, and write it back. This is the only
    # RCNN pass — the classifier then runs on the box-cropped triplet (no second detection).
    ref_depth = rcnn_roi.reference_depth(patient_ts)

    def bbox_provider(index, images):
        if ref_depth is not None and patient_ts.has_rcnn_box(ref_depth, index):
            return patient_ts.get_rcnn_box(ref_depth, index)
        bbox = eb.detect_triplet_bbox(images)
        if ref_depth is not None:
            patient_ts.set_rcnn_box(
                ref_depth, index, list(bbox) if bbox is not None else None, save=False
            )
        return bbox

    flushed_count = len(raw_by_tp)

    def persist() -> Dict[str, Any]:
        # Flush the shared ROI boxes alongside the scores so both survive an interruption.
        patient_ts.save_rcnn_boxes()
        return _persist_predictions(
            patient_ts, raw_by_tp, num_timepoints, class_names, labeler_names, model_name, fingerprint
        )

    def result_cb(index: int, scores: np.ndarray) -> None:
        nonlocal flushed_count
        raw_by_tp[index] = [float(v) for v in np.asarray(scores, dtype=np.float64).ravel()]
        if len(raw_by_tp) - flushed_count >= PRED_FLUSH_EVERY:
            persist()
            flushed_count = len(raw_by_tp)
        if progress_cb is not None:
            progress_cb(len(raw_by_tp), num_timepoints)

    eb.run_timelapse_inference(
        triplets,
        model_name,
        use_gpu=use_gpu,
        done_indices=set(raw_by_tp),
        bbox_provider=bbox_provider,
        result_cb=result_cb,
        should_cancel=should_cancel,
        should_yield=should_yield,
    )

    # Final durable flush — persists whatever was scored, even if cancelled part-way, so
    # the next run resumes from here rather than recomputing. Nothing scored (immediate
    # cancel, no resume) leaves the sidecar untouched.
    return persist() if raw_by_tp else patient_ts.load_predictions()


# --------------------------------------------------------------------------------------
# Reading predictions back for the timeline / anomaly layers.
# --------------------------------------------------------------------------------------
def load_prediction_arrays(patient_ts) -> Optional[Dict[str, Any]]:
    """Return per-timepoint prediction arrays (index 0..T-1), or None if absent.

    Keys: ``class_names``, ``raw_class``/``raw_class_index``,
    ``post_class``/``post_class_index``, ``max_prob`` (lists indexed by timepoint),
    and ``num_timepoints``.
    """
    predictions = patient_ts.load_predictions()
    timepoints = predictions.get("timepoints") if isinstance(predictions, dict) else None
    if not timepoints:
        return None

    indices = sorted(int(key) for key in timepoints)
    num_timepoints = (max(indices) + 1) if indices else 0

    raw_class: List[Optional[str]] = [None] * num_timepoints
    raw_class_index: List[Optional[int]] = [None] * num_timepoints
    post_class: List[Optional[str]] = [None] * num_timepoints
    post_class_index: List[Optional[int]] = [None] * num_timepoints
    max_prob: List[Optional[float]] = [None] * num_timepoints

    for key, entry in timepoints.items():
        timepoint = int(key)
        raw_class[timepoint] = entry.get("raw_class")
        raw_class_index[timepoint] = entry.get("raw_class_index")
        post_class[timepoint] = entry.get("post_class")
        post_class_index[timepoint] = entry.get("post_class_index")
        max_prob[timepoint] = entry.get("max_prob")

    return {
        "class_names": predictions.get("class_names", []),
        "raw_class": raw_class,
        "raw_class_index": raw_class_index,
        "post_class": post_class,
        "post_class_index": post_class_index,
        "max_prob": max_prob,
        "num_timepoints": num_timepoints,
    }


def predictions_stale(patient_ts) -> bool:
    """True if predictions exist but were computed against a different image set."""
    if not patient_ts.has_predictions():
        return False
    metadata = patient_ts.load_prediction_metadata()
    stored = metadata.get("image_fingerprint")
    if stored is None:
        return False
    return stored != patient_ts.image_fingerprint()


def predictions_complete(patient_ts) -> bool:
    """True only when every timepoint has been scored and persisted.

    A partially-computed sidecar (an interrupted/resumable run) returns False so callers
    re-launch the worker to resume. Legacy sidecars written before incremental support
    (no ``complete`` flag) are treated as complete.
    """
    if not patient_ts.has_predictions():
        return False
    metadata = patient_ts.load_prediction_metadata()
    if "complete" in metadata:
        return bool(metadata["complete"])
    return True
