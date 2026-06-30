# Labeling Acceleration Plan

## Goal

Turn the labeler from a fully manual tool into an assisted one: preload per-timepoint
**OCR hours** and **model stage predictions**, surface **anomalies** that likely need
correction, and let the user accept or fix labels with minimal keystrokes — while keeping
manual scroll-and-label as a first-class fallback.

**Definition of done:** A user can load a patient, see a populated timeline within seconds,
jump directly to stage boundaries and anomalies, and label a full patient mostly by
accepting predictions and correcting boundaries (few clicks, keyboard-driven).

## Constraints (from `CLAUDE.md`)

- Priority order: labeling speed/keyboard UX → durable label writes → lazy image loading.
- Persistence and image-path logic live in `data.py`. No Qt widgets in `data.py`.
- Keep image loading lazy unless the user explicitly opts into preloading.
- Do **not** modify data in `embpred-data` or `embpred_deploy`.

## Non-goals
- Training or fine-tuning models.
- Changing the existing `labels.json` / `label_metadata.json` schema for human labels
  (predictions go in separate sidecar files).

## External dependencies

Located in /Users/berk/code
- `embpred_deploy` — model inference + AWS weight download.
  Docs: see its `README.md` (Model Weights Installation, Timelapse Inference, `--postprocess`).
- `embpred-data` — read-only test datasets for manual verification.
- `Notebook_OCR.ipynb` — reference RapidOCR pipeline for time extraction.

---

## Key integration risks (resolve before/while building)

These are concrete mismatches found in the current code that must be handled explicitly,
ideally in a dedicated mapping/adapter layer rather than ad hoc in `gui.py`.

### Focal depth mismatch
- Labeler uses 7 depths (`F-45 … F45`, `ORDERED_FOCAL_DEPTH`).
- Deploy ResNet inference uses **3 depths** (`F-15, F0, F15`) with RCNN bbox extraction.
- Plan: run inference on the 3-depth subset; keep all 7 for labeling/best-depth UI.

### `get_ROI()` is a placeholder
- `get_ROI()` in `gui.py` is currently a center crop. It should instead produce a true embryo
  ROI via `embpred_deploy`'s Faster-RCNN bbox detection (`ExtractEmbFrame` /
  `extract_emb_frame_2d` in `embpred_deploy/rcnn.py`, driven by `rcnn.pt`). See Phase 4.
- This shares the RCNN model/weights with Phase 3 inference, so load the model once and reuse.

### Required weight artifacts
- Inference needs the main `*.pth` checkpoint **and** `rcnn.pt` in `MODELS_DIR`.
- The AWS setup step must fetch **all** required artifacts, not just the ResNet checkpoint.

### Weight download mechanism
- `embpred_deploy` README documents `aws s3 cp` from private bucket `cfai-model-weights`.
- `install_weights.py` only unzips a local archive — it does not download from S3.
- Decision: add a thin `setup_models.py` that resolves `embpred_deploy.config.MODELS_DIR`,
  checks for required files, and runs `aws s3 cp` for missing ones (auth via user's AWS config).

### Performance / threading
- PyTorch + inference is heavy. Run OCR and inference **off the UI thread** with a progress
  indicator; never block labeling. Honors the lazy-loading priority.

---

## Persistence: per-patient sidecar files

Computed artifacts live next to each patient's images, separate from human labels.

| File | Shape | Contents |
| --- | --- | --- |
| `ocr_times.json` | `{ "<t>": hours }` | OCR-extracted time in hours per timepoint; `null` on failure |
| `predictions.json` | per-timepoint | raw class scores + postprocessed class index/name |
| `rcnn_boxes.json` | `{ "<depth>": { "<t>": [x1,y1,x2,y2] } }` | RCNN embryo bbox per (depth, timepoint); `null` when detection fell back to center crop |
| `prediction_metadata.json` | object | model name, RCNN weights name, embpred version, timestamp, depth subset used |

Rules:
- Compute once per patient; recompute only on explicit "Re-run predictions" or when the
  image set changes (detect via count/mtime).
- `labels.json` remains human ground truth only; predictions never overwrite it.

---

## Phased delivery

### Phase 0 — Discovery & spec
- Read `embpred_deploy` (`main.py`, `post_process.py`, `config.py`, `install_weights.py`).
- Run `embpred_deploy --timelapse-dir ... --postprocess` on one `embpred-data` patient to
  capture expected outputs (`raw_timelapse_outputs.npy/csv`, `max_prob_classes.csv`,
  `postprocessed_timelapse_outputs.*`).
- Deliverable: documented **class mapping table** and **depth mapping** decisions.

### Phase 1 — Model weight setup (AWS)
- `setup_models.py`: locate `MODELS_DIR`, list required files (`{model}.pth`, `rcnn.pt`),
  download missing ones via `aws s3 cp` for AWS-authenticated users.
- On app startup: check presence; if missing, prompt the user (don't auto-block labeling of
  already-labelable patients).
- Walk user through auth in the UI if needed
- Acceptance: fresh env → run setup → `MODELS_DIR` contains all required artifacts.

### Phase 2 — OCR timestamps (`ocr.py`)
- Port from `Notebook_OCR.ipynb`: crop a small bottom-right ROI scaled to image size
  (notebook uses ~44×44 px; offsets differ for 500×500 vs 800×800), run RapidOCR on a
  reference depth (F0), parse hours via regex `(\d+(?:\.\d+)?)\s*h`.
- Failures → interpolate from nearby frames and flag
- Should be run all at once in a seperate thread as specified above
- Acceptance: plot hours vs timepoint index; spot-check 5 frames against displayed time.

### Phase 3 — Inference + postprocess cache (`predictions.py`)
- Wrap `embpred_deploy` inference for the 3-depth subset; apply `monotonic_decoding`
  (postprocess) for the strict stage sequence.
- Map deploy classes → labeler classes; write `predictions.json` + `prediction_metadata.json`.
- Acceptance: cached postprocessed sequence matches the CLI `--postprocess` output for the
  same patient.

### Phase 4 — Replace `get_ROI()` with RCNN bbox extraction
- Replace the placeholder center crop in `gui.py`'s `get_ROI()` with embryo bounding-box
  detection using `embpred_deploy`'s Faster-RCNN: `ExtractEmbFrame` / `extract_emb_frame_2d`
  in `embpred_deploy/rcnn.py`, loaded from `rcnn.pt` in `MODELS_DIR`.
- Wrap the RCNN call in a backend module (`rcnn_roi.py`, or extend `predictions.py`) so model
  logic stays out of `gui.py`; `get_ROI()` just calls the wrapper.
- Reuse the **same** loaded RCNN model/device as Phase 3 inference (load once, share) — this is
  the same detector that produces the inference ROI.
- Fall back to the current center crop when no box is found (mirror embpred's `fallback_size`).
- Cache detected boxes in the `rcnn_boxes.json` sidecar (per depth + timepoint) so scrolling
  reuses them instead of re-running the detector; recompute only on demand or when the image
  set changes. `null` entries record fall-to-center-crop cases. Keep loading lazy.
- Acceptance: ROI crops track the embryo across timepoints on an `embpred-data` patient, with
  graceful fallback when detection fails.

### Phase 5 — Combined timeline UI
- Timeline widget: x-axis = OCR hours, stage shown as colored band / y-position.
- Overlay predicted vs saved label; default to postprocessed sequence, toggle raw argmax.
- Click a segment or marker → jump to that timepoint (reuse existing Left/Right navigation).
- Reuse matplotlib patterns from `DashboardWindow`.

### Phase 6 — Anomaly surfacing & stage auto-fill
- Anomaly signals (computed in `anomalies.py`, surfaced on the timeline):

  | Signal | Meaning |
  | --- | --- |
  | Low `max(prob)` at a timepoint | uncertain prediction |
  | Raw argmax ≠ postprocessed | monotonic decode changed the label |
  | Large jump in OCR hours | missing/extra frames |
  | Prediction ≠ existing `labels.json` | candidate for relabeling |
  | Stage run length < N frames | suspiciously short stage |

- Keyboard-first **stage auto-fill**:
  - Mark stage start `[` and end `]` on the timeline.
  - "Fill stage" applies the chosen class to indices `[start, end]` via `data.py` label APIs.
  - "Accept prediction for stage" fills from postprocessed boundaries.
- Navigation: "Next anomaly" / "Previous anomaly" shortcuts; highlight stage boundaries as
  primary review targets.

---

## Proposed module map

- `ocr.py` — crop, RapidOCR, hour parsing.
- `predictions.py` — embpred inference wrapper, class mapping, sidecar cache I/O.
- `rcnn_roi.py` — Faster-RCNN bbox wrapper (`ExtractEmbFrame`); shared by inference and
  `get_ROI()`. (May live inside `predictions.py` to share the loaded model.)
- `anomalies.py` — scoring from predictions + OCR + labels.
- `setup_models.py` — AWS weight presence check/download.
- `data.py` — load/save sidecar JSON (no Qt).
- `gui.py` — timeline widget, progress dialogs, shortcuts (no persistence/inference logic).

---

## Test plan (manual, on read-only `embpred-data`)

1. **Weights:** fresh env → setup → `MODELS_DIR` has required files.
2. **OCR:** plot hours vs timepoint; spot-check 5 frames.
3. **Inference:** cached postprocessed CSV matches CLI `--postprocess`.
4. **ROI:** `get_ROI()` returns RCNN-detected crops that track the embryo across timepoints;
   falls back to center crop when detection fails.
5. **Timeline:** load patient, jump to anomaly, edit a label, reload — cache and
   `labels.json` stay consistent.
6. **Auto-fill:** fill a stage from `[` to `]`; verify writes via `labels.json`.

---

## Open questions
- Default model name to fetch and use: 
  - answer: aws list and allow user to select. embpred_deploy handles mapping
- Re-run predictions automatically when weights update, or only on demand?
  - answer: only on demand
- Which depth is the OCR reference (assume F0)?
  - answer: F0 but retry with other depths if failed
- Anomaly threshold values (`max(prob)` cutoff, minimum stage length `N`).
