"""Adapter around the external ``embpred_deploy`` package.

This is the single module in the labeler that imports ``embpred_deploy`` and, by
extension, the heavy ML stack (torch, torchvision, opencv). Everything torch-related
is imported lazily *inside functions* so that merely importing this module — or the
GUI — stays cheap and never drags in PyTorch at startup.

It also owns the in-process model cache. The Faster-RCNN detector is loaded once and
shared between Phase 3 inference (``run_timelapse_inference``) and the Phase 4
``get_ROI`` bbox extraction (``detect_bbox``), exactly as ``plan.md`` requires
("load once, share").

``embpred_deploy`` is expected to be importable. If it is not installed into the
active environment we fall back to the sibling source checkout at
``../embpred_deploy`` (the layout on this machine). We never modify anything inside
that repo — we only read its code and weights.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Constants that do NOT require torch (safe to read at import time).
# --------------------------------------------------------------------------------------

# Mirrors ``class_mapping`` in embpred_deploy/main.py. Index order is the developmental
# progression that ``monotonic_decoding`` relies on. Verified against embpred at runtime
# (see ``deploy_class_mapping``). 14 classes.
DEPLOY_CLASS_MAPPING: Dict[int, str] = {
    0: "t1",
    1: "tPN",
    2: "tPNf",
    3: "t2",
    4: "t3",
    5: "t4",
    6: "t5",
    7: "t6",
    8: "t7",
    9: "t8",
    10: "tM",
    11: "tB",
    12: "tEB",
    13: "tEmpty",
}
DEPLOY_CLASS_NAMES: List[str] = [DEPLOY_CLASS_MAPPING[i] for i in range(len(DEPLOY_CLASS_MAPPING))]
NCLASS = len(DEPLOY_CLASS_MAPPING)

# Deploy ResNet inference uses a 3-depth subset; these names match the labeler's
# ORDERED_FOCAL_DEPTH entries so we can pull the right image folders directly.
INFERENCE_DEPTHS: List[str] = ["F-15", "F0", "F15"]

RCNN_FILENAME = "rcnn.pt"
SIZE = (224, 224)

# Sibling source checkout used only if embpred_deploy is not pip-installed.
_SIBLING_EMBPRED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "embpred_deploy"))

_import_lock = threading.Lock()
_model_lock = threading.Lock()
# Serializes model forward passes: the RCNN detector is shared between the bbox and
# prediction background tasks, and torch modules aren't guaranteed thread-safe for
# concurrent forward calls. (It also avoids CPU thread oversubscription.)
_forward_lock = threading.Lock()
_embpred_ready = False

# Cached models: load once, reuse across calls and threads.
_rcnn_cache: Optional[Tuple[object, object]] = None  # (model, device)
_classifier_cache: Dict[str, Tuple[object, object]] = {}  # name -> (model, device)


class EmbpredUnavailableError(RuntimeError):
    """Raised when embpred_deploy / its ML stack cannot be imported."""


# --------------------------------------------------------------------------------------
# Locating and importing embpred_deploy.
# --------------------------------------------------------------------------------------
def _ensure_embpred_importable() -> None:
    """Make ``import embpred_deploy`` work, falling back to the sibling checkout."""
    global _embpred_ready
    if _embpred_ready:
        return
    with _import_lock:
        if _embpred_ready:
            return
        try:
            import embpred_deploy  # noqa: F401
        except ImportError:
            if os.path.isdir(os.path.join(_SIBLING_EMBPRED_DIR, "embpred_deploy")):
                if _SIBLING_EMBPRED_DIR not in sys.path:
                    sys.path.insert(0, _SIBLING_EMBPRED_DIR)
            else:
                raise EmbpredUnavailableError(
                    "Could not import 'embpred_deploy' and no sibling checkout was found at "
                    f"{_SIBLING_EMBPRED_DIR}. Install embpred_deploy (and torch/torchvision/"
                    "opencv) into this environment."
                )
        _embpred_ready = True


def _import_torch():
    """Import torch lazily, turning ML-stack import failures into a clear error."""
    _ensure_embpred_importable()
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise EmbpredUnavailableError(
            "PyTorch is required for inference/ROI detection but could not be imported. "
            "Install torch + torchvision (see embpred_deploy README)."
        ) from exc


# --------------------------------------------------------------------------------------
# Weight-file discovery (torch-free).
# --------------------------------------------------------------------------------------
def models_dir() -> str:
    """Absolute path of embpred_deploy's MODELS_DIR (where weights live)."""
    _ensure_embpred_importable()
    from embpred_deploy.config import MODELS_DIR  # lightweight (pathlib only)

    return str(MODELS_DIR)


def rcnn_path() -> str:
    return os.path.join(models_dir(), RCNN_FILENAME)


def classifier_path(model_name: str) -> str:
    return os.path.join(models_dir(), f"{model_name}.pth")


def list_local_models() -> List[str]:
    """Names of classifier checkpoints (``*.pth``) present in MODELS_DIR."""
    try:
        directory = models_dir()
    except EmbpredUnavailableError:
        return []
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.splitext(name)[0]
        for name in os.listdir(directory)
        if name.endswith(".pth")
    )


def rcnn_present() -> bool:
    try:
        return os.path.exists(rcnn_path())
    except EmbpredUnavailableError:
        return False


def required_files(model_name: str) -> List[str]:
    """Filenames that must exist in MODELS_DIR to run inference with this model."""
    return [f"{model_name}.pth", RCNN_FILENAME]


def missing_files(model_name: str) -> List[str]:
    try:
        directory = models_dir()
    except EmbpredUnavailableError:
        return required_files(model_name)
    return [name for name in required_files(model_name) if not os.path.exists(os.path.join(directory, name))]


def embpred_version() -> str:
    try:
        _ensure_embpred_importable()
        import embpred_deploy

        return getattr(embpred_deploy, "__version__", "unknown")
    except Exception:
        return "unknown"


def deploy_class_mapping() -> Dict[int, str]:
    """The authoritative deploy index->name mapping.

    Uses embpred's own mapping when importable (source of truth) and verifies our
    hardcoded copy matches; otherwise returns the hardcoded copy.
    """
    try:
        _ensure_embpred_importable()
        from embpred_deploy.main import class_mapping as deploy_cm
    except (EmbpredUnavailableError, ImportError):
        # Package not found, or its ML deps (cv2/torch) aren't installed in this
        # environment: fall back to the verified hardcoded copy.
        return dict(DEPLOY_CLASS_MAPPING)

    if dict(deploy_cm) != DEPLOY_CLASS_MAPPING:
        # Surface drift loudly rather than silently mismapping predictions.
        raise RuntimeError(
            "embpred_deploy class_mapping has changed and no longer matches "
            "embpred_backend.DEPLOY_CLASS_MAPPING; update the labeler mapping.\n"
            f"  embpred: {dict(deploy_cm)}\n  labeler: {DEPLOY_CLASS_MAPPING}"
        )
    return dict(deploy_cm)


# --------------------------------------------------------------------------------------
# Model loading / cache.
# --------------------------------------------------------------------------------------
def _resolve_device(use_gpu: Optional[bool]):
    torch = _import_torch()
    if use_gpu is None:
        from embpred_deploy.utils import get_device

        return get_device()
    return torch.device("cuda") if (use_gpu and torch.cuda.is_available()) else torch.device("cpu")


def get_rcnn_model(use_gpu: Optional[bool] = None):
    """Load (and cache) the Faster-RCNN detector. Returns ``(model, device)``."""
    global _rcnn_cache
    with _model_lock:
        if _rcnn_cache is not None:
            return _rcnn_cache
        _ensure_embpred_importable()
        from embpred_deploy.main import load_faster_RCNN_model_device

        device = _resolve_device(use_gpu)
        path = rcnn_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"RCNN weights not found at {path}. Run model setup first.")
        model, model_device = load_faster_RCNN_model_device(path, use_GPU=(device.type == "cuda"))
        _rcnn_cache = (model, model_device)
        return _rcnn_cache


def get_classifier_model(model_name: str, use_gpu: Optional[bool] = None):
    """Load (and cache) a classification checkpoint. Returns ``(model, device)``."""
    with _model_lock:
        cached = _classifier_cache.get(model_name)
        if cached is not None:
            return cached
        _ensure_embpred_importable()
        from embpred_deploy.models.mapping import model_mapping
        from embpred_deploy.utils import load_model

        if model_name not in model_mapping:
            raise KeyError(
                f"Model '{model_name}' is not in embpred_deploy's model_mapping. "
                f"Available: {sorted(model_mapping.keys())}"
            )
        path = classifier_path(model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model weights not found at {path}. Run model setup first.")

        device = _resolve_device(use_gpu)
        model_class, class_args = model_mapping[model_name][0], model_mapping[model_name][1]
        model, _epoch, _auc = load_model(
            path, device, NCLASS, model_class=model_class, class_args=class_args
        )
        _classifier_cache[model_name] = (model, device)
        return _classifier_cache[model_name]


def clear_model_cache() -> None:
    """Drop cached models (e.g. after downloading new weights)."""
    global _rcnn_cache
    with _model_lock:
        _rcnn_cache = None
        _classifier_cache.clear()


# --------------------------------------------------------------------------------------
# Inference.
# --------------------------------------------------------------------------------------
def run_timelapse_inference(
    triplet_paths: Sequence[Tuple[str, str, str]],
    model_name: str,
    use_gpu: Optional[bool] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """Run the 3-depth ResNet over a timelapse and return raw scores ``(T, NCLASS)``.

    ``triplet_paths[t]`` is ``(F-15_path, F0_path, F15_path)`` for timepoint ``t``.
    Mirrors the embpred CLI's ``--timelapse-dir`` path (grayscale reads, RCNN bbox
    extraction, ``inference(..., map_output=False)``) so the cached postprocessed
    sequence matches ``embpred_deploy --postprocess`` for the same patient.
    """
    _ensure_embpred_importable()
    import cv2
    from embpred_deploy.main import inference

    # Touch the mapping check so a drifted embpred mapping fails fast.
    deploy_class_mapping()

    classifier, device = get_classifier_model(model_name, use_gpu=use_gpu)
    rcnn_model, _rcnn_device = get_rcnn_model(use_gpu=use_gpu)

    total = len(triplet_paths)
    outputs: List[np.ndarray] = []
    for index, paths in enumerate(triplet_paths):
        if should_cancel is not None and should_cancel():
            break
        images = []
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Failed to read image for inference: {path}")
            images.append(img)
        with _forward_lock:
            raw = inference(
                classifier,
                device,
                images,
                map_output=False,
                output_to_str=False,
                rcnn_model=rcnn_model,
                size=SIZE,
            )
        outputs.append(np.asarray(raw, dtype=np.float64))
        if progress_cb is not None:
            progress_cb(index + 1, total)

    if not outputs:
        return np.empty((0, NCLASS), dtype=np.float64)
    result = np.vstack(outputs)
    if result.shape[1] != NCLASS:
        # The labeler's class mapping + monotonic decoding assume the 14-class head.
        raise ValueError(
            f"Model '{model_name}' produced {result.shape[1]} classes but the labeler "
            f"expects {NCLASS}. Choose a {NCLASS}-class model (e.g. a CustomResNet50)."
        )
    return result


def postprocess_monotonic(raw_outputs: np.ndarray) -> np.ndarray:
    """Apply embpred's monotonic decoding -> class index per timepoint ``(T,)``."""
    _ensure_embpred_importable()
    from embpred_deploy.post_process import monotonic_decoding

    raw_outputs = np.asarray(raw_outputs, dtype=np.float64)
    if raw_outputs.ndim != 2 or raw_outputs.shape[0] == 0:
        return np.empty((0,), dtype=int)
    return np.asarray(monotonic_decoding(raw_outputs, loss="NLL"), dtype=int)


# --------------------------------------------------------------------------------------
# RCNN bounding-box detection (shared with get_ROI).
# --------------------------------------------------------------------------------------
def _to_uint8_gray(image: np.ndarray) -> np.ndarray:
    """Coerce an arbitrary-dtype 2D/3D image to a uint8 grayscale array.

    The labeler reads tiffs that may be uint16 or RGB; the detector and cv2 colour
    conversion expect single-channel uint8.
    """
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    arr = arr.astype(np.float64)
    max_val = float(arr.max()) if arr.size else 0.0
    if max_val > 255.0:  # e.g. uint16 -> scale into 0..255
        arr = arr / max_val * 255.0
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


def detect_bbox(
    image_2d: np.ndarray, use_gpu: Optional[bool] = None
) -> Optional[Tuple[int, int, int, int]]:
    """Detect the embryo bounding box in a single 2D image.

    Returns ``(x_min, y_min, x_max, y_max)`` in the coordinates of ``image_2d``, or
    None when the detector finds nothing (caller should fall back to a center crop).
    """
    _ensure_embpred_importable()
    from embpred_deploy.rcnn import get_emb_frame_bbox

    rcnn_model, device = get_rcnn_model(use_gpu=use_gpu)
    gray = _to_uint8_gray(image_2d)
    with _forward_lock:
        bbox = get_emb_frame_bbox(gray, rcnn_model, device)
    if bbox is None:
        return None
    x_min, y_min, x_max, y_max = (int(v) for v in bbox)
    return x_min, y_min, x_max, y_max
