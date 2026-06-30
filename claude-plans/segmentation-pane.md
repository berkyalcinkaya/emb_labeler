# Segmentation Pane Plan

## Goal

Extend the labeler beyond per-timepoint classification to **pixel-level segmentation** at two
specific embryo stages, reusing the timepoint predictions/labels the app already produces to get
the user to the right frames fast.

- **tPN stage** — segment the **pronucleus** (find-and-segment task).
- **tB (blastocyst) stage** — segment three structures: **trophectoderm (TE)**, **inner cell mass
  (ICM)**, and **zona pellucida (ZP)**. Treat this as a per-pixel multi-class problem (background +
  the three structures).

Segmentation lives in a **separate pane/mode** in the UI, distinct from the timepoint-labeling
view. Drawing must be fast and keyboard-driven, modeled on `yeastvision`'s pyqtgraph painting.

**Definition of done:** From a loaded patient with existing timepoint predictions, the user can
jump directly to the tPN and tB frames, open the segmentation pane, paint/erase masks for the
relevant structures with a responsive brush, and have those masks persist durably to per-patient
sidecar files that survive a crash and reload exactly as drawn. File formats should be easily usable to train a segmentation model downstream. 

## Scope & non-goals

- **No** segmentation model, inference, or auto-segmentation in the UI. This is a **manual drawing
  tool only**. (Predicted timepoints are used purely for navigation.)
- **No** retraining, fine-tuning, or model-weight handling for segmentation. However, eventually, we 
- **Do not** change the human timepoint-label schema (`labels.json` / `label_metadata.json`) or the
  existing computed-artifact sidecars (`predictions.json`, `ocr_times.json`, `rcnn_boxes.json`).
  Segmentation masks are a **new, separate** artifact.
- **Do not** modify data in `embpred-data` or `embpred_deploy`.

## Where this fits in the existing app

You are building on a working labeler. Before designing, read these to ground your decisions:

- `CLAUDE.md` — architecture boundaries and priorities (labeling speed > durable writes > lazy
  loading). **Persistence stays in `data.py`; Qt widgets stay in `gui.py`. Do not cross these.**
- `data.py` — how the app loads patient image series across focal depths, and the durable
  sidecar pattern already in use: atomic temp-file-then-`os.replace` JSON writes
  (`_write_json_atomic`), tolerant reads that fall back to a default on corruption (`_read_json`),
  and the image-set fingerprint (`image_fingerprint`) used to detect stale artifacts.
- `gui.py` — the labeling window, class definitions (`CLASSES`, including `tPN` and `tB`), hotkey
  conventions, the dark theme palette, navigation (Left/Right = timepoint, Up/Down = focal depth),
  and how timepoint predictions are read and shown.
- `predictions.py` — how per-timepoint stage predictions are produced/cached, so you can use them
  to locate the tPN and tB frames.

Reuse these existing patterns rather than inventing parallel ones.

## Reference implementation: yeastvision drawing

The user explicitly wants the fast-drawing approach drawn from `yeastvision`
(`/Users/berk/code/yeastvision`). Study and adapt — do not copy wholesale:

- `yeastvision/yeastvision/parts/canvas.py`:
  - `ImageDraw(pg.ImageItem)` — a pyqtgraph image item that paints label values directly into a
    numpy mask array via `drawAt()`, using a precomputed brush kernel (`setDrawKernel`,
    adjustable brush size). The colored overlay is produced by indexing a colormap
    (`maskColors[currMask]`). This is the core pattern for responsive painting.
  - `ViewBoxNoRightDrag` — a `pg.ViewBox` subclass that repurposes mouse buttons (e.g. so one
    button paints and another erases/pans). Adapt the interaction model to our needs.
  - Note the kernel-clipping logic in `drawAt` that keeps strokes inside image bounds.
- Take inspiration from how yeastvision keeps the mask as a compact integer-labeled array (one
  value per class) and renders it as a translucent color overlay on top of the grayscale image.

## Persistence requirements (durable, fault-tolerant, fast)

Follow the same philosophy as yeastvision's mask storage and the app's existing sidecars:

- Masks are a **new per-patient computed artifact**, stored next to the patient's images, separate
  from human timepoint labels. **You decide the on-disk format and filenames** — but the choice
  must satisfy:
  - **Durable / fault-tolerant:** a crash mid-write must not corrupt previously saved masks
    (mirror the atomic-write pattern already in `data.py`).
  - **Fast:** saving a stroke or frame must not stutter the brush; loading must not block the UI.
    Per-frame integer label arrays (not RGB) are the expected representation — pick an efficient
    encoding (e.g. compressed array files, run-length, or similar) and justify it briefly.
  - **Self-describing:** record which structures/class indices a mask uses, the image dimensions
    it was drawn against, and the stage (tPN vs tB) so masks can be validated against the image on
    reload. Detect staleness with the existing `image_fingerprint` mechanism.
- All file path resolution and read/write logic goes in `data.py` (no Qt there). The pane in
  `gui.py` calls those APIs; it never touches the filesystem directly.
- A mask must round-trip exactly: draw → save → reload → identical pixels.

## Navigation: get the user to tPN and tB fast

This is a primary UX requirement, not an afterthought. The app already produces per-timepoint
stage predictions; use them so the user lands on the right frames with minimal effort.

- Provide one-action jumps to the first predicted **tPN** and **tB** frames (keyboard shortcut and/or a
  control in the pane). 
- If predictions are missing or the patient has no predicted tPN or tB, surface this to the user in an error upon trying to click on the tPN or tB navigation button. 
-  Keep manual scroll-and-label fully functional as a fallback.
- Show depth F-0 by default for segmentation but allow scrolloing through focal depths like in yeastvision and in the timepoint labeler. 

## UX conventions to honor

- Keyboard-first and minimal clicks (consistent with the rest of the app).
- Per-structure selection with single-key hotkeys (like the existing label chips), a brush-size
  control, and an eraser. Use the existing dark theme palette and assign distinct, legible overlay
  colors per structure (TE / ICM / ZP / pronucleus).
- Translucent overlay so the underlying embryo image stays visible while painting.
- Make it obvious which stage/structure is active and whether a frame already has a saved mask.

## Decisions delegated to you (the implementing LLM)

The user has intentionally left technical decisions open. **Make and document them**; do not block
on asking. Prefer the smallest design consistent with the constraints above. You decide:

1. **Mask storage format & filenames** (encoding, one-file-per-stage vs per-frame, compression).
2. **Pane architecture** — a separate window vs an embedded mode/tab; how it shares patient/image
   state with the main labeling view.
3. **Drawing internals** — how much of yeastvision's `ImageDraw`/`ViewBox` to adapt vs rebuild,
   brush model, erase model, undo/redo (at minimum, support correcting mistakes).
4. **Class/structure model** — index assignment for background + structures per stage, and how the
   two stages' label sets are represented.
5. **Focal-depth handling** in the pane and which depth(s) segmentation targets.
6. **Navigation specifics** — exact shortcuts, multi-frame stage handling, missing-prediction
   fallback.
7. **Save cadence** — autosave-on-stroke vs explicit save vs debounced, balancing the
   speed-vs-durability priority order.

## Additional Notes and Assumptions
- Should a frame allow overlapping structures, or is each pixel exactly one class (expected:
  exactly one class per pixel at tB; pronucleus vs background at tPN)?
  - exactly one class per pixel
- Is segmentation done on a single focal depth, or should masks be tracked per depth?
  - answer: masks are per-timepoint meaning a the same segmentation masks corresponds to all focal depths at a given tiimepoint. 

## Implementation decisions (resolved)

The delegated decisions, as built (see `data.py` / `gui.py` docstrings for detail):

1. **Storage format & filenames.** One compressed file per saved mask:
   `<patient>/segmentation/<stage>_t<NNNN>.npz` (zero-padded timepoint). Each holds a
   `uint8 (H, W)` label array (`mask`) plus an embedded JSON `meta` blob (format version,
   stage, dims, class index→name, `image_fingerprint`, timestamp) — self-describing and
   directly trainable via `np.load(p)["mask"]`. Per-frame files keep stroke saves cheap
   and crash-isolated; writes are atomic (temp + `os.replace`, via `_write_npz_atomic`).
   An all-background mask deletes its file so "has mask" stays accurate.
2. **Pane architecture.** Separate `SegmentationWindow(QMainWindow)`, matching the
   existing `AllDepthsWindow` / `TimelineWindow` pattern; the app holds a reference and
   repoints it via `set_patient` on patient change. Keeps the labeling cockpit untouched
   and lets the pane own a custom painting `ViewBox`.
3. **Drawing internals.** Adapted from yeastvision's `ImageDraw` / `ViewBox`: row-major
   mask, disk brush stamped (clipped) along the dragged line. **Left-drag paints,
   right-drag pans, wheel zooms.** Eraser = brush with class 0. Undo/redo via bounded
   pre-stroke mask snapshots (simple + exact). Mass cleanup mirrors yeastvision's
   click-select-then-delete: **Ctrl+click** (or the `⊙ select` / `V` mode) highlights the
   8-connected component under the cursor (toggle to add/remove), and **Backspace/Delete**
   erases the selected masses (undoable); since a class value can span disconnected blobs
   here, selection is by connected component (`scipy.ndimage.label`), not by class value.
4. **Class model.** `data.SEGMENTATION_STAGES` (domain) defines indices: `tPN = [bg,
   pronucleus]`, `tB = [bg, TE, ICM, ZP]`. Colors/hotkeys live in `gui.py` (UI). One
   class per pixel.
5. **Focal depth.** Mask is per-timepoint (depth-agnostic). F0 default; ↑↓ scrolls depths
   and only swaps the background image, leaving the overlay intact.
6. **Navigation.** `⤓ tPN` / `⤓ tB` (and `P` / `B`) jump to the first postprocessed-
   prediction frame of that stage; missing predictions or no such stage surface a dialog.
   Arrow keys remain a manual fallback.
7. **Save cadence.** Debounced autosave (~600 ms) on stroke-end, plus a synchronous flush
   before any navigation / stage switch / patient swap / window close.
