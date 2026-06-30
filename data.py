"""Backend data model for the embryo labeling GUI.

Expected directory structure
----------------------------
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

Each PatientTimeSeries stores image paths lazily. Images are only loaded when
requested, unless load=True is passed.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from skimage.io import imread

ORDERED_FOCAL_DEPTH = ["F-45", "F-30", "F-15", "F0", "F15", "F30", "F45"]
IMAGE_EXTENSIONS = (".png", ".jpeg", ".jpg", ".tif", ".tiff")

_RUN_RE = re.compile(r"RUN(\d+)")


def run_sort_key(path: str) -> Tuple[Union[int, float], str]:
    """Order frames by their ``RUN<n>`` index (true temporal order).

    Mirrors embpred_deploy's ``sort_embryo_fname_by_run`` so the labeler's timepoint
    indices line up with embpred's timelapse inference / ``--postprocess`` output. For
    zero-padded filenames this yields the same order as a plain lexical sort, so
    already-labeled patients with padded names are unaffected; only non-padded
    ``RUN`` numbers (where lexical puts ``RUN10`` before ``RUN2``) are reordered.

    Files without a ``RUN`` token sort last; the filename is a deterministic
    tie-breaker (embpred relies on os.listdir order for ties, which isn't stable).
    """
    filename = os.path.basename(path)
    match = _RUN_RE.search(filename)
    run_index: Union[int, float] = int(match.group(1)) if match else float("inf")
    return (run_index, filename)

# Computed-artifact sidecar files, written next to each patient's images and kept
# separate from the human-authored labels.json / label_metadata.json. The schema of
# each file is owned by the module that produces it (ocr.py, predictions.py,
# rcnn_roi.py); data.py only provides path resolution and durable load/save.
OCR_TIMES_FILENAME = "ocr_times.json"
PREDICTIONS_FILENAME = "predictions.json"
RCNN_BOXES_FILENAME = "rcnn_boxes.json"
PREDICTION_METADATA_FILENAME = "prediction_metadata.json"


class DataSet:
    """A top-level dataset containing many PatientTimeSeries objects."""

    def __init__(self, root_directory: str):
        self.root_directory = os.path.abspath(root_directory)
        self.patient_series_list: List[PatientTimeSeries] = []
        self.load_patient_series()

    def load_patient_series(self) -> None:
        self.patient_series_list.clear()
        patient_dirs = [
            os.path.join(self.root_directory, subdir)
            for subdir in sorted(os.listdir(self.root_directory))
            if os.path.isdir(os.path.join(self.root_directory, subdir))
        ]
        for patient_dir in patient_dirs:
            self.patient_series_list.append(PatientTimeSeries(patient_dir))

    def print_all_patient_series(self) -> None:
        for patient_ts in self.patient_series_list:
            print(f"Patient Directory: {patient_ts.directory}")
            shape = patient_ts.get_shape()
            if shape:
                print(f"Data Shape: {shape}")
            else:
                print("Data not loaded or no images found.")

    def load_all_data(self) -> None:
        for patient_ts in self.patient_series_list:
            patient_ts.load_data()

    def get_patient_series(self) -> List["PatientTimeSeries"]:
        return self.patient_series_list

    def num_patients(self) -> int:
        return len(self.patient_series_list)

    def num_timepoints(self) -> int:
        return sum(patient.num_timepoints() for patient in self.patient_series_list)


class PatientTimeSeries:
    """A lazy representation of one patient's embryo time series.

    `image_paths` is depth-major:
        image_paths[depth_index][timepoint_index] -> image path

    If loaded into memory, `data` has shape:
        (t, r, c, d)
    """

    def __init__(self, directory: str, load: bool = False):
        self.directory = os.path.abspath(directory)
        self.patient_id = os.path.basename(os.path.normpath(self.directory))
        self.depths = ORDERED_FOCAL_DEPTH
        self.loaded = load
        self.data: Optional[np.ndarray] = None
        self.image_paths = self.get_image_paths()
        self.labels = self.load_labels()
        self.label_metadata = self.load_label_metadata()
        # Computed-artifact sidecars are loaded lazily so building a DataSet (which
        # constructs every PatientTimeSeries up front) stays cheap.
        self._rcnn_boxes: Optional[Dict[str, Dict[str, Any]]] = None
        if load:
            self.load_data()

    def get_image_paths(self) -> List[List[str]]:
        image_paths: List[List[str]] = []
        for depth_name in self.depths:
            depth_dir = os.path.join(self.directory, depth_name)
            if not os.path.isdir(depth_dir):
                image_paths.append([])
                continue

            paths = [
                os.path.join(depth_dir, filename)
                for filename in os.listdir(depth_dir)
                if filename.lower().endswith(IMAGE_EXTENSIONS)
            ]
            # Order by RUN index (true temporal order), matching embpred's sort so
            # predictions/OCR align with the labeler's timepoint indices.
            paths.sort(key=run_sort_key)
            image_paths.append(paths)
        return image_paths

    def load_data(self) -> None:
        if not self.image_paths or not self.image_paths[0]:
            raise ValueError(f"No images found for patient: {self.directory}")

        depth_stacks = []
        for depth_paths in self.image_paths:
            stack = [self.load_image(image_path) for image_path in depth_paths]
            depth_stacks.append(np.stack(stack, axis=0))  # (t, r, c)
        self.data = np.stack(depth_stacks, axis=-1)       # (t, r, c, d)
        self.loaded = True

    def load_image(self, path: str) -> np.ndarray:
        return imread(path)

    def get_shape(self) -> Optional[tuple[int, int, int, int]]:
        if self.data is not None:
            return self.data.shape

        if not self.image_paths or not self.image_paths[0]:
            return None

        first_image = self.load_image(self.image_paths[0][0])
        t = self.num_timepoints()
        r, c = first_image.shape[:2]
        d = self.num_depths()
        return (t, r, c, d)

    def get_image(self, timepoint: int, depth: int) -> np.ndarray:
        self._validate_timepoint_depth(timepoint, depth)
        if self.data is None:
            return self.load_image(self.image_paths[depth][timepoint])
        return self.data[timepoint, :, :, depth]

    def load_focal_depths(self, timepoint: int) -> np.ndarray:
        if timepoint < 0 or timepoint >= self.num_timepoints():
            raise IndexError("Timepoint index out of range.")
        if self.data is None:
            depth_images = [imread(depth_paths[timepoint]) for depth_paths in self.image_paths]
            return np.stack(depth_images, axis=-1)  # (r, c, d)
        return self.data[timepoint]

    def get_depth_name(self, depth_index: int) -> str:
        return self.depths[depth_index]

    def num_depths(self) -> int:
        return len(self.depths)

    def depth_has_images(self, depth_index: int) -> bool:
        return 0 <= depth_index < len(self.image_paths) and len(self.image_paths[depth_index]) > 0

    def populated_depth_indices(self) -> List[int]:
        """Indices of depths that actually have images (some patients, e.g. the
        3-depth inference sets, only populate a subset of ORDERED_FOCAL_DEPTH)."""
        return [i for i in range(self.num_depths()) if self.depth_has_images(i)]

    def default_depth_index(self) -> int:
        """A sensible starting depth: F0 if present, else the first populated depth."""
        populated = self.populated_depth_indices()
        if not populated:
            return 0
        if "F0" in self.depths and self.depths.index("F0") in populated:
            return self.depths.index("F0")
        return populated[0]

    def num_timepoints(self) -> int:
        if self.data is not None:
            return int(self.data.shape[0])
        if not self.image_paths:
            return 0
        # Use the shortest depth stack so indexing is safe even if folders differ.
        nonempty_lengths = [len(paths) for paths in self.image_paths if paths]
        return min(nonempty_lengths) if nonempty_lengths else 0

    def _validate_timepoint_depth(self, timepoint: int, depth: int) -> None:
        if depth < 0 or depth >= self.num_depths():
            raise IndexError("Depth index out of range.")
        if timepoint < 0 or timepoint >= self.num_timepoints():
            raise IndexError("Timepoint index out of range.")
        if self.data is None and timepoint >= len(self.image_paths[depth]):
            raise IndexError("Timepoint index out of range for this focal depth.")

    # -------------------------
    # Label persistence
    # -------------------------
    def labels_path(self) -> str:
        return os.path.join(self.directory, "labels.json")

    def metadata_path(self) -> str:
        return os.path.join(self.directory, "label_metadata.json")

    def load_labels(self) -> Dict[str, Dict[str, Any]]:
        path = self.labels_path()
        if not os.path.exists(path):
            return {"timepoint_labels": {}, "best_depths": {}}

        with open(path, "r", encoding="utf-8") as file:
            loaded = json.load(file)

        # Migration path for older labels.json files that were flat timepoint->label dicts.
        if "timepoint_labels" not in loaded and "best_depths" not in loaded:
            return {"timepoint_labels": loaded, "best_depths": {}}

        loaded.setdefault("timepoint_labels", {})
        loaded.setdefault("best_depths", {})
        return loaded

    def save_labels(self) -> None:
        with open(self.labels_path(), "w", encoding="utf-8") as file:
            json.dump(self.labels, file, indent=2, sort_keys=True)

    def load_label_metadata(self) -> List[Dict[str, Any]]:
        path = self.metadata_path()
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []

    def save_label_metadata(self) -> None:
        with open(self.metadata_path(), "w", encoding="utf-8") as file:
            json.dump(self.label_metadata, file, indent=2, sort_keys=True)

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _append_metadata_event(
        self,
        *,
        event_type: str,
        timepoint: int,
        value: Any,
        previous_value: Any,
        action: str,
        complete: bool = True,
    ) -> None:
        self.label_metadata.append(
            {
                "timestamp": self._now_iso(),
                "patient_id": self.patient_id,
                "patient_directory": self.directory,
                "event_type": event_type,
                "timepoint": int(timepoint),
                "value": value,
                "previous_value": previous_value,
                "action": action,
                "complete": bool(complete),
            }
        )

    def set_label(self, timepoint: int, label: str) -> None:
        key = str(timepoint)
        previous = self.labels["timepoint_labels"].get(key)
        action = "created" if previous is None else "updated"
        if previous == label:
            return

        self.labels["timepoint_labels"][key] = label
        self._append_metadata_event(
            event_type="timepoint_label",
            timepoint=timepoint,
            value=label,
            previous_value=previous,
            action=action,
            complete=True,
        )
        self.save_labels()
        self.save_label_metadata()

    def get_label(self, timepoint: int) -> Optional[str]:
        return self.labels["timepoint_labels"].get(str(timepoint))

    def set_best_depths(self, timepoint: int, depths: List[str]) -> None:
        if len(depths) > 3:
            raise ValueError("At most three focal depths can be selected.")
        invalid = [depth for depth in depths if depth not in self.depths]
        if invalid:
            raise ValueError(f"Invalid focal depth names: {invalid}")

        key = str(timepoint)
        previous = self.labels["best_depths"].get(key)
        if previous == depths:
            return

        previous_complete = isinstance(previous, list) and len(previous) == 3
        complete = len(depths) == 3
        if previous is None:
            action = "created" if complete else "partial_created"
        elif previous_complete and not complete:
            action = "partial_updated"
        elif not previous_complete and complete:
            action = "created"
        else:
            action = "updated"

        self.labels["best_depths"][key] = depths
        self._append_metadata_event(
            event_type="best_depths",
            timepoint=timepoint,
            value=depths,
            previous_value=previous,
            action=action,
            complete=complete,
        )
        self.save_labels()
        self.save_label_metadata()

    def get_best_depths(self, timepoint: int) -> List[str]:
        value = self.labels["best_depths"].get(str(timepoint), [])
        return value if isinstance(value, list) else []

    # -------------------------
    # Computed-artifact sidecar persistence
    #
    # These files hold OCR times, model predictions, and RCNN boxes. They are
    # written separately from labels.json (human ground truth) and never overwrite
    # it. Each file's inner schema is defined by its producing module; data.py only
    # resolves paths and reads/writes JSON durably.
    # -------------------------
    def ocr_times_path(self) -> str:
        return os.path.join(self.directory, OCR_TIMES_FILENAME)

    def predictions_path(self) -> str:
        return os.path.join(self.directory, PREDICTIONS_FILENAME)

    def rcnn_boxes_path(self) -> str:
        return os.path.join(self.directory, RCNN_BOXES_FILENAME)

    def prediction_metadata_path(self) -> str:
        return os.path.join(self.directory, PREDICTION_METADATA_FILENAME)

    @staticmethod
    def _read_json(path: str, default: Any) -> Any:
        if not os.path.exists(path):
            return default
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (json.JSONDecodeError, OSError):
            return default

    @staticmethod
    def _write_json_atomic(path: str, obj: Any) -> None:
        """Write JSON via a temp file + os.replace so a crash mid-write cannot
        leave a truncated sidecar (priority: durable writes)."""
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as file:
            json.dump(obj, file, indent=2, sort_keys=True)
        os.replace(tmp_path, path)

    # --- OCR times -----------------------------------------------------
    def load_ocr_times(self) -> Dict[str, Any]:
        loaded = self._read_json(self.ocr_times_path(), {})
        return loaded if isinstance(loaded, dict) else {}

    def save_ocr_times(self, ocr_times: Dict[str, Any]) -> None:
        self._write_json_atomic(self.ocr_times_path(), ocr_times)

    # --- Predictions ---------------------------------------------------
    def load_predictions(self) -> Dict[str, Any]:
        loaded = self._read_json(self.predictions_path(), {})
        return loaded if isinstance(loaded, dict) else {}

    def save_predictions(self, predictions: Dict[str, Any]) -> None:
        self._write_json_atomic(self.predictions_path(), predictions)

    def load_prediction_metadata(self) -> Dict[str, Any]:
        loaded = self._read_json(self.prediction_metadata_path(), {})
        return loaded if isinstance(loaded, dict) else {}

    def save_prediction_metadata(self, metadata: Dict[str, Any]) -> None:
        self._write_json_atomic(self.prediction_metadata_path(), metadata)

    def has_predictions(self) -> bool:
        return os.path.exists(self.predictions_path())

    # --- RCNN boxes (read/written incrementally as the user scrolls) ---
    def _ensure_rcnn_boxes_loaded(self) -> Dict[str, Dict[str, Any]]:
        if self._rcnn_boxes is None:
            loaded = self._read_json(self.rcnn_boxes_path(), {})
            self._rcnn_boxes = loaded if isinstance(loaded, dict) else {}
        return self._rcnn_boxes

    def has_rcnn_box(self, depth_name: str, timepoint: int) -> bool:
        """True when a box has been computed for this (depth, timepoint).

        Note a computed-but-failed detection is stored as ``null`` and still
        returns True here — callers use ``get_rcnn_box`` to read the value.
        """
        depth_map = self._ensure_rcnn_boxes_loaded().get(depth_name)
        return isinstance(depth_map, dict) and str(timepoint) in depth_map

    def get_rcnn_box(self, depth_name: str, timepoint: int) -> Optional[List[int]]:
        """Return the cached ``[x1, y1, x2, y2]`` box, or None when detection
        fell back to a center crop (or when nothing is cached yet — guard with
        ``has_rcnn_box`` to distinguish the two)."""
        depth_map = self._ensure_rcnn_boxes_loaded().get(depth_name)
        if not isinstance(depth_map, dict):
            return None
        value = depth_map.get(str(timepoint))
        return list(value) if isinstance(value, list) else None

    def set_rcnn_box(
        self, depth_name: str, timepoint: int, box: Optional[List[int]], save: bool = True
    ) -> None:
        boxes = self._ensure_rcnn_boxes_loaded()
        depth_map = boxes.setdefault(depth_name, {})
        depth_map[str(timepoint)] = [int(v) for v in box] if box is not None else None
        if save:
            self.save_rcnn_boxes()

    def save_rcnn_boxes(self) -> None:
        if self._rcnn_boxes is not None:
            self._write_json_atomic(self.rcnn_boxes_path(), self._rcnn_boxes)

    def clear_rcnn_boxes(self) -> None:
        """Drop cached boxes from memory and disk (used when re-detecting)."""
        self._rcnn_boxes = {}
        if os.path.exists(self.rcnn_boxes_path()):
            os.remove(self.rcnn_boxes_path())

    # --- Image-set fingerprint (detect when computed artifacts are stale) ---
    def image_fingerprint(self) -> Dict[str, List[int]]:
        """A cheap signature of the image set: per depth, ``[count, latest_mtime]``.

        Stored alongside predictions so they can be flagged stale and recomputed
        when images are added/removed/replaced.
        """
        fingerprint: Dict[str, List[int]] = {}
        for depth_name, paths in zip(self.depths, self.image_paths):
            if not paths:
                continue
            try:
                latest_mtime = max(int(os.path.getmtime(path)) for path in paths)
            except OSError:
                latest_mtime = 0
            fingerprint[depth_name] = [len(paths), latest_mtime]
        return fingerprint
