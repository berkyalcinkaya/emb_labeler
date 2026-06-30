# Embryo Labeler

Desktop GUI for labeling time-lapse embryo images across patients, timepoints, and focal depths.

## Priorities (in order)
1. Labeling speed and keyboard-driven UX
2. Correct, durable writes to patient `labels.json` / `label_metadata.json`
3. Keep image loading lazy unless explicitly preloading

## Stack
- Python 3, PyQt5, pyqtgraph, numpy, scikit-image, matplotlib
- Run: `mamba activate <env>` then `python gui.py`
- Deps: `requirements.txt`

## Architecture
- `data.py` — `DataSet`, `PatientTimeSeries`, label I/O, lazy image loading
- `gui.py` — `EmbryoLabelingApp`, `AllDepthsWindow`, `DashboardWindow`
- `Notebook_OCR.ipynb` — reference for RapidOCR time extraction (planned feature)

Do not put persistence or image-path logic in `gui.py`. Do not put Qt widgets in `data.py`.

## Domain
- **Focal depths** (fixed order): F-45 … F45 (`ORDERED_FOCAL_DEPTH` in `data.py`)
- **Timepoint labels**: tEmpty, t2–t8, tPN, tPNf, tM, tBlastocyst (`CLASSES` in `gui.py`)
- **Best depths**: subset of focal depths per timepoint
- Outputs per patient: `labels.json`, `label_metadata.json` (see README for schema)

## UX conventions
- Arrow keys: Left/Right = timepoint, Up/Down = focal depth
- Drag dataset root onto window to load
- New UI for labeling should prefer shortcuts and minimal clicks
- `get_ROI()` is being replaced: it should return an embryo ROI from `embpred_deploy`'s
  Faster-RCNN bbox detection (`rcnn.pt`), with a center-crop fallback. Boxes are cached in a
  per-patient sidecar file. See `plan.md` Phase 4.

## Workflow
- Use a mamba environment for all Python work
- New features on a branch; one commit per completed feature
- No test suite yet — manually verify with a small patient folder

## Related repos (do not modify their data)
- `embpred_deploy` — model inference / AWS weight download (future dependency)
- `embpred-data` — test datasets only

## Roadmap
See `plan.md` for prediction preload, OCR timeline, RCNN ROI detection, anomaly surfacing, and
stage auto-fill. Computed artifacts (OCR times, predictions, RCNN boxes) are cached in
per-patient sidecar files, kept separate from human-authored `labels.json`.