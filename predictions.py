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
# --------------------------------------------------------------------------------------
def compute_and_save_predictions(
    patient_ts,
    model_name: str,
    use_gpu: Optional[bool] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Run inference for ``patient_ts``, persist sidecars, and return the predictions.

    Writes ``predictions.json`` and ``prediction_metadata.json`` via the patient's
    durable sidecar I/O. Never touches ``labels.json``. Returns the predictions dict
    (also useful directly by the caller); returns ``{}`` if cancelled before any
    timepoint completed.
    """
    triplets = _inference_triplet_paths(patient_ts)
    class_names = [eb.DEPLOY_CLASS_MAPPING[i] for i in range(eb.NCLASS)]

    raw_outputs = eb.run_timelapse_inference(
        triplets,
        model_name,
        use_gpu=use_gpu,
        progress_cb=progress_cb,
        should_cancel=should_cancel,
    )
    # Don't persist a partial/empty run (e.g. the user cancelled mid-way).
    if raw_outputs.shape[0] == 0 or (should_cancel is not None and should_cancel()):
        return {}

    post_indices = eb.postprocess_monotonic(raw_outputs)

    timepoints: Dict[str, Any] = {}
    for timepoint in range(raw_outputs.shape[0]):
        scores = raw_outputs[timepoint]
        probs = _softmax(scores)
        raw_index = int(np.argmax(scores))
        post_index = int(post_indices[timepoint])
        timepoints[str(timepoint)] = {
            "raw_scores": [float(v) for v in scores],
            "raw_class_index": raw_index,
            "raw_class": map_deploy_to_labeler(class_names[raw_index]),
            "post_class_index": post_index,
            "post_class": map_deploy_to_labeler(class_names[post_index]),
            "max_prob": float(probs.max()),
        }

    predictions = {
        "class_names": [map_deploy_to_labeler(name) for name in class_names],
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
        "num_timepoints": int(raw_outputs.shape[0]),
        "image_fingerprint": patient_ts.image_fingerprint(),
    }

    patient_ts.save_predictions(predictions)
    patient_ts.save_prediction_metadata(metadata)
    return predictions


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
