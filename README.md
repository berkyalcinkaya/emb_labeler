# Embryo Labeler

A PyQt5 + pyqtgraph desktop GUI for labeling time-lapse embryo images across patients, timepoints, and focal depths.

## Expected dataset structure

```text
root_dataset/
  patient_001/
    F-45/*.tif
    F-30/*.tif
    F-15/*.tif
    F0/*.tif
    F15/*.tif
    F30/*.tif
    F45/*.tif
  patient_002/
    ...
```

Depth folders are read in this order:

```python
["F-45", "F-30", "F-15", "F0", "F15", "F30", "F45"]
```

Within each depth, frames are ordered by their `RUN<n>` index (matching
embpred_deploy's sort), so timepoint indices reflect true temporal order and line up
with embpred's timelapse inference / `--postprocess` output. For zero-padded filenames
this matches a plain lexical sort; files without a `RUN` token fall back to filename
order.

## Run

```bash
pip install -r requirements.txt
python gui.py
```

Drag a dataset root folder onto the window (or **File ▸ Open Dataset…**).

## Interface

The window is a keyboard-first "cockpit" (dark theme):

- **Header** — patient selector, focal-depth nav (`◂ F0 ▸`), the `t / N` timepoint
  counter, view toggles (ROI inset, Auto-ROI, All depths), and three compact
  background-task dots (OCR / Pred / ROI); click an active dot to cancel it.
- **Main image** — fills the pane with no axis/histogram chrome; the **ROI** is a
  picture-in-picture inset in the corner (toggle with the header **ROI** checkbox).
- **HUD bar** (under the image) — current timepoint, OCR hour, model prediction +
  confidence, your saved label, and any anomaly for this frame.
- **Timeline strip** (bottom) — the embedded stage/OCR timeline; click to jump, toggle
  raw/postprocessed, or `⤢` to open the full-size window.
- **Right rail** — label chips (each shows its hotkey; the model's predicted class is
  outlined in orange) and best-depth chips (the depth you're viewing is outlined white).

### Labeling

Press a class's hotkey to label the current timepoint — `0` = tEmpty, `1`–`8` = t1–t8,
and `Q W E R T` = tPN, tPNf, tM, tEB, tB (or click the chip). Arrow keys navigate:
**← →** timepoint, **↑ ↓** focal depth. Best depths (up to 3) are toggled with their
chips.

## Files written into each patient folder

### `labels.json`

```json
{
  "timepoint_labels": {
    "0": "tPN",
    "1": "t2"
  },
  "best_depths": {
    "0": ["F-15", "F0", "F15"]
  }
}
```

### `label_metadata.json`

A list of label-save events, used by the Dashboard to show labeling progress over time.

```json
[
  {
    "timestamp": "2026-05-27T12:34:56+00:00",
    "patient_id": "patient_001",
    "event_type": "timepoint_label",
    "timepoint": 0,
    "value": "tPN",
    "previous_value": null,
    "action": "created",
    "complete": true
  }
]
```

## Dashboard

Open **Tools > Dashboard** after loading a dataset. It shows:

1. Completion percentages for timepoint labels and best-depth labels, overall and by patient.
2. Distribution of timepoint classes across the dataset.
3. Cumulative labeling progress over time from `label_metadata.json`.

## Assisted labeling

The labeler can preload **OCR hours** and **model stage predictions**, surface
**anomalies**, and let you accept/fill labels with few keystrokes. These features are
optional — if their dependencies (see `requirements.txt`) are missing, the core labeler
still works (center-crop ROIs, no predictions).

### Automatic background processing

When you select a patient, three tasks start automatically off the UI thread, each shown
as a colored dot in the header: **OCR**, **Pred** (predictions), and **ROI** (RCNN
bboxes). A dot is gray when idle, green while running/done, and red on error (hover for
detail); click an active dot to cancel it. Already-current results are skipped
(OCR/predictions recompute only when missing or when the image set changed; bbox skips
frames already cached), and labeling is never blocked. Predictions default to the model
`New-ResNet50-Unfreeze-CE-embSplits-…-3layer256,128,64`; if its weights aren't present
the **Pred** dot turns red with "weights missing" — fetch it (or pick another model) via
**Tools ▸ Setup Models**. The Tools menu can also re-trigger any of the three on demand.

### Tools menu

- **Setup Models…** — choose a local classifier checkpoint, or fetch the list from the
  private S3 bucket (`cfai-model-weights`) and download missing weights via the AWS CLI.
  Both the chosen `*.pth` checkpoint and `rcnn.pt` are required.
- **Run OCR Times** — OCRs the embedded clock (bottom-right of each frame, reference
  depth F0 with retries) and writes per-timepoint hours; failures are interpolated.
- **Run / Re-run Predictions** — runs inference on the 3-depth subset (`F-15, F0, F15`),
  applies monotonic postprocessing, and caches per-timepoint predictions. Re-run
  recomputes; predictions are also flagged when the image set changes.
- **Detect ROIs (RCNN)** — batch-detects embryo bounding boxes (current depth / the
  3 inference depths / all depths) and caches them.
- **Timeline** — stage/OCR timeline (x = OCR hours, y = stage). Shows the postprocessed
  sequence (toggle raw argmax), saved labels, and flagged anomalies. Click to jump.

### ROI detection

`EmbryoLabelingApp.get_ROI()` now uses embpred_deploy's Faster-RCNN (`rcnn.pt`) with a
center-crop fallback. With **Auto-detect ROI** on (default), the ROI view shows an
instant center crop and is replaced by the detected box once detection finishes for that
frame (off the UI thread); boxes are cached so revisiting is instant.

### Keyboard shortcuts (assisted)

| Key | Action |
| --- | --- |
| `[` / `]` | Mark stage start / end at the current timepoint |
| `F` | Fill the marked stage `[start, end]` with the selected class |
| `A` | Accept the prediction at the current timepoint |
| `N` / `P` | Jump to next / previous flagged anomaly |

(Arrow keys still drive timepoint/depth navigation.) The **Assist** menu also exposes
"Accept predictions for stage" and "Toggle raw / postprocessed".

## Files written into each patient folder (computed artifacts)

These sidecars are written next to the images and are kept **separate** from the
human-authored `labels.json` / `label_metadata.json` (which are never overwritten):

| File | Contents |
| --- | --- |
| `ocr_times.json` | OCR-extracted hours per timepoint (+ which were interpolated) |
| `predictions.json` | Per-timepoint raw scores, raw argmax, and postprocessed class |
| `rcnn_boxes.json` | RCNN embryo box per `(depth, timepoint)`; `null` = center-crop fallback |
| `prediction_metadata.json` | Model name, RCNN weights, embpred version, timestamp, depth subset, image fingerprint |

## Notes

The human-label `CLASSES` list previously contained a `"tEB" "tB"` typo (missing comma)
that Python concatenated into a single bogus `"tEBtB"` class; this is fixed, and the 14
labeler classes now align 1:1 by name with embpred_deploy's predicted classes.
