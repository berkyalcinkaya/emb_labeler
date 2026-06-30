"""Anomaly scoring over a patient's predictions, OCR hours, and existing labels.

Pure functions (no Qt, no torch). The GUI uses the output to mark the timeline and to
drive Next/Previous-anomaly navigation. Stage boundaries (where the postprocessed stage
changes) are surfaced separately as primary review targets.

Signals (thresholds are module constants, tune as needed):
  - low_confidence : softmax max prob below LOW_PROB_THRESHOLD
  - decode_changed : raw argmax != monotonic-decoded class
  - ocr_jump       : OCR-hour gap to the previous frame is unusually large
  - pred_mismatch  : postprocessed prediction differs from the saved human label
  - short_stage    : a postprocessed stage runs for fewer than MIN_STAGE_LENGTH frames
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import ocr
import predictions as predmod

LOW_PROB_THRESHOLD = 0.5
MIN_STAGE_LENGTH = 2
OCR_JUMP_FACTOR = 2.5

SIGNAL_LOW_CONFIDENCE = "low_confidence"
SIGNAL_DECODE_CHANGED = "decode_changed"
SIGNAL_OCR_JUMP = "ocr_jump"
SIGNAL_PRED_MISMATCH = "pred_mismatch"
SIGNAL_SHORT_STAGE = "short_stage"


def _stage_runs(class_indices: List[Optional[int]]) -> List[Dict[str, int]]:
    """Contiguous runs of equal class index: ``[{start, end, length, cls}, ...]``."""
    runs: List[Dict[str, int]] = []
    start = None
    for i, value in enumerate(class_indices):
        if value is None:
            if start is not None:
                runs.append({"start": start, "end": i - 1, "length": i - start, "cls": class_indices[start]})
                start = None
            continue
        if start is None:
            start = i
        elif class_indices[i] != class_indices[start]:
            runs.append({"start": start, "end": i - 1, "length": i - start, "cls": class_indices[start]})
            start = i
    if start is not None:
        n = len(class_indices)
        runs.append({"start": start, "end": n - 1, "length": n - start, "cls": class_indices[start]})
    return runs


def stage_boundaries(post_class_index: List[Optional[int]]) -> List[int]:
    """Timepoints where the postprocessed stage starts (incl. the first known one)."""
    boundaries: List[int] = []
    previous: Optional[int] = None
    for i, value in enumerate(post_class_index):
        if value is None:
            continue
        if previous is None or value != previous:
            boundaries.append(i)
        previous = value
    return boundaries


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def compute_anomalies(
    patient_ts,
    low_prob_threshold: float = LOW_PROB_THRESHOLD,
    min_stage_length: int = MIN_STAGE_LENGTH,
    ocr_jump_factor: float = OCR_JUMP_FACTOR,
) -> Dict[str, Any]:
    """Score anomalies for a patient. Returns a structure with per-timepoint signals,
    a sorted list of flagged timepoints, and stage boundaries.

    Returns ``num_timepoints == 0`` style empties when no predictions exist yet.
    """
    pred = predmod.load_prediction_arrays(patient_ts)
    if pred is None:
        return {"num_timepoints": 0, "per_timepoint": [], "flagged": [], "boundaries": [], "signal_counts": {}}

    num_timepoints = pred["num_timepoints"]
    raw_index = pred["raw_class_index"]
    post_index = pred["post_class_index"]
    post_class = pred["post_class"]
    max_prob = pred["max_prob"]

    ocr_data = ocr.load_hours_array(patient_ts)
    hours: List[Optional[float]] = ocr_data["hours"] if ocr_data else [None] * num_timepoints
    if len(hours) < num_timepoints:
        hours = hours + [None] * (num_timepoints - len(hours))

    signals: List[List[Dict[str, Any]]] = [[] for _ in range(num_timepoints)]

    # Low confidence + decode changed.
    for t in range(num_timepoints):
        prob = max_prob[t] if t < len(max_prob) else None
        if prob is not None and prob < low_prob_threshold:
            signals[t].append({"type": SIGNAL_LOW_CONFIDENCE, "detail": f"max prob {prob:.2f}"})
        if raw_index[t] is not None and post_index[t] is not None and raw_index[t] != post_index[t]:
            signals[t].append(
                {
                    "type": SIGNAL_DECODE_CHANGED,
                    "detail": f"raw {pred['raw_class'][t]} -> decoded {post_class[t]}",
                }
            )

    # OCR jumps relative to the typical inter-frame gap.
    gaps = []
    for t in range(1, num_timepoints):
        if hours[t] is not None and hours[t - 1] is not None:
            gaps.append(hours[t] - hours[t - 1])
    positive_gaps = [g for g in gaps if g > 0]
    median_gap = _median(positive_gaps)
    if median_gap > 0:
        for t in range(1, num_timepoints):
            if hours[t] is not None and hours[t - 1] is not None:
                gap = hours[t] - hours[t - 1]
                if gap > ocr_jump_factor * median_gap:
                    signals[t].append(
                        {"type": SIGNAL_OCR_JUMP, "detail": f"+{gap:.1f}h (median {median_gap:.1f}h)"}
                    )

    # Prediction vs saved label.
    for t in range(num_timepoints):
        label = patient_ts.get_label(t)
        if label is not None and post_class[t] is not None and label != post_class[t]:
            signals[t].append(
                {"type": SIGNAL_PRED_MISMATCH, "detail": f"predicted {post_class[t]} vs label {label}"}
            )

    # Suspiciously short postprocessed stages.
    for run in _stage_runs(post_index):
        if run["length"] < min_stage_length:
            for t in range(run["start"], run["end"] + 1):
                signals[t].append(
                    {"type": SIGNAL_SHORT_STAGE, "detail": f"{post_class[t]} lasts {run['length']} frame(s)"}
                )

    per_timepoint = []
    flagged: List[int] = []
    signal_counts: Dict[str, int] = {}
    for t in range(num_timepoints):
        if signals[t]:
            flagged.append(t)
            for signal in signals[t]:
                signal_counts[signal["type"]] = signal_counts.get(signal["type"], 0) + 1
        per_timepoint.append({"timepoint": t, "signals": signals[t], "score": len(signals[t])})

    return {
        "num_timepoints": num_timepoints,
        "per_timepoint": per_timepoint,
        "flagged": flagged,
        "boundaries": stage_boundaries(post_index),
        "signal_counts": signal_counts,
    }
