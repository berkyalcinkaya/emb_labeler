"""PyQt5 + pyqtgraph desktop GUI for embryo timepoint labeling."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pyqtgraph.Qt import QtCore
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QKeySequence, QPalette
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import anomalies
import ocr as ocr_module
import predictions as predmod
import rcnn_roi
import setup_models
from data import DataSet, ORDERED_FOCAL_DEPTH, SEGMENTATION_STAGES

# Human-label classes. These are intentionally the same 14 names embpred_deploy
# predicts (see embpred_backend.DEPLOY_CLASS_MAPPING), so predictions map onto labels
# 1:1 by name. NOTE: the trailing entries were previously "tEB" "tB" with a missing
# comma, which Python silently concatenated into a single bogus "tEBtB" class; the
# comma below fixes that.
CLASSES = [
    "tEmpty",
    "t1",
    "t2",
    "t3",
    "t4",
    "t5",
    "t6",
    "t7",
    "t8",
    "tPN",
    "tPNf",
    "tM",
    "tEB",
    "tB",
]

# Default classifier for automatic predictions. May not be present locally — if its
# weights are missing the prediction task reports it and the user can fetch it via
# Tools > Setup Models (or pick another model there).
DEFAULT_MODEL = "New-ResNet50-Unfreeze-CE-embSplits-overUnderSampleMedian-lessregularized-nodropout-3layer256,128,64"

# Single-key hotkeys for each label class (shown on the chips). Digits 0-8 map to the
# cleavage stages (0 = empty, 1..8 = t1..t8); the QWERT row covers the morphological
# stages. None of these collide with the navigation/assist keys ([ ] arrows F A N P).
LABEL_HOTKEYS: Dict[str, str] = {
    "tEmpty": "0",
    "t1": "1", "t2": "2", "t3": "3", "t4": "4",
    "t5": "5", "t6": "6", "t7": "7", "t8": "8",
    "tPN": "Q", "tPNf": "W", "tM": "E", "tEB": "R", "tB": "T",
}

# ----------------------------------------------------------------------------
# Cohesive dark theme. The image viewers were already black, so theming the
# surrounding chrome to match makes the whole window read as one surface.
# ----------------------------------------------------------------------------
C_BG = "#1e1e1e"        # window background
C_PANEL = "#252526"     # raised panels (rail, header, HUD, timeline)
C_PANEL2 = "#2d2d30"    # controls (chips, buttons)
C_HOVER = "#3e3e42"
C_BORDER = "#3a3a3d"
C_TEXT = "#e0e0e0"
C_MUTED = "#9a9a9a"
C_ACCENT = "#2ca02c"    # current selection / saved label
C_PRED = "#ff8c00"      # model prediction
C_ANOM = "#e05252"      # anomaly
C_IMG_BG = "#141414"    # image letterbox (near-black, blends with embryo frame)

STYLESHEET = f"""
QMainWindow, QWidget {{ background: {C_BG}; color: {C_TEXT}; }}
QMenuBar {{ background: {C_PANEL}; color: {C_TEXT}; }}
QMenuBar::item:selected {{ background: {C_HOVER}; }}
QMenu {{ background: {C_PANEL}; color: {C_TEXT}; border: 1px solid {C_BORDER}; }}
QMenu::item:selected {{ background: {C_HOVER}; }}
QToolTip {{ background: {C_PANEL2}; color: {C_TEXT}; border: 1px solid {C_BORDER}; }}
#Header, #Hud, #Rail, #TimelineBar {{ background: {C_PANEL}; }}
#Header {{ border-bottom: 1px solid {C_BORDER}; }}
#TimelineBar {{ border-top: 1px solid {C_BORDER}; }}
#RailTitle {{ color: {C_MUTED}; font-size: 10px; font-weight: 600; letter-spacing: 1px; }}
#PiP {{ border: 1px solid {C_BORDER}; background: {C_IMG_BG}; }}
QComboBox {{
    background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 5px;
    padding: 3px 8px; min-height: 20px;
}}
QComboBox:hover {{ border-color: {C_ACCENT}; }}
QComboBox QAbstractItemView {{
    background: {C_PANEL2}; color: {C_TEXT}; selection-background-color: {C_ACCENT};
}}
QScrollArea {{ border: none; }}
/* Background-task progress bars (rail). The chunk color tracks task state. */
QProgressBar {{
    background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 3px;
    min-height: 6px; max-height: 6px;
}}
QProgressBar::chunk {{ background: {C_MUTED}; border-radius: 3px; }}
QProgressBar[state="running"]::chunk,
QProgressBar[state="done"]::chunk {{ background: {C_ACCENT}; }}
QProgressBar[state="error"] {{ border-color: {C_ANOM}; }}
QProgressBar[state="error"]::chunk {{ background: {C_ANOM}; }}
QScrollBar:vertical {{ background: {C_BG}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {C_BORDER}; border-radius: 5px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
/* Generic flat buttons (header view toggles, nav arrows) */
QToolButton {{
    background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 5px;
    padding: 3px 8px; color: {C_TEXT};
}}
QToolButton:hover {{ background: {C_HOVER}; }}
QToolButton:checked {{ background: {C_ACCENT}; border-color: {C_ACCENT}; color: #0c1f0c; }}
/* Label chips */
QPushButton[chip="label"] {{
    text-align: left; padding: 5px 9px; border-radius: 6px;
    border: 1px solid {C_BORDER}; background: {C_PANEL2}; color: {C_TEXT};
}}
QPushButton[chip="label"]:hover {{ background: {C_HOVER}; }}
QPushButton[chip="label"]:checked {{
    background: {C_ACCENT}; border-color: {C_ACCENT}; color: #0c1f0c; font-weight: 600;
}}
QPushButton[chip="label"][predicted="true"] {{ border: 2px solid {C_PRED}; }}
"""


def apply_dark_palette(app: QApplication) -> None:
    """Fusion + dark palette so native menus/dialogs match the QSS-styled widgets."""
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(C_BG))
    palette.setColor(QPalette.WindowText, QColor(C_TEXT))
    palette.setColor(QPalette.Base, QColor(C_PANEL))
    palette.setColor(QPalette.AlternateBase, QColor(C_PANEL2))
    palette.setColor(QPalette.Text, QColor(C_TEXT))
    palette.setColor(QPalette.Button, QColor(C_PANEL2))
    palette.setColor(QPalette.ButtonText, QColor(C_TEXT))
    palette.setColor(QPalette.ToolTipBase, QColor(C_PANEL2))
    palette.setColor(QPalette.ToolTipText, QColor(C_TEXT))
    palette.setColor(QPalette.Highlight, QColor(C_ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#0c1f0c"))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(C_MUTED))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(C_MUTED))
    app.setPalette(palette)


class TaskProgressBar(QWidget):
    """A labeled progress bar for one background task (OCR / predictions / ROI boxes).

    Lives in the right rail and is a *view*: the app owns the per-embryo workers and
    drives the bar with :meth:`set_running` / :meth:`set_done` / :meth:`set_idle` /
    :meth:`set_error`. Idle / running / done / error are conveyed by the bar's fill color
    and status text; an errored task (e.g. missing weights) turns the bar red.

    Clicking invokes the handler set with :meth:`set_click_handler`; the app interprets it
    for the *current* embryo's task — running → pause, paused/idle → resume, error → retry,
    done → no-op. Pausing is the only way to stop a task; navigation never does. The
    tooltip always states what a click will do.
    (The legacy ``start``/``worker`` path is still used by the dataset-wide prefetch bar,
    which drives its single worker's progress directly.)
    """

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._name = name
        # Single click handler wired by the app. The app owns the (per-embryo) worker and
        # decides what a click does based on the task's current state, so the bar is just a
        # view + a click surface. No-op until set.
        self._click_handler: Optional[Callable[[], None]] = None

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(4)
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet(f"color: {C_TEXT}; font-size: 10px; font-weight: 600;")
        self.status_label = QLabel("idle")
        self.status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        head.addWidget(self.name_label)
        head.addStretch(1)
        head.addWidget(self.status_label)
        layout.addLayout(head)

        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        layout.addWidget(self.bar)

        self.worker: Optional[FunctionWorker] = None
        self.setCursor(Qt.PointingHandCursor)
        self._set_state("idle", "idle")

    # --- state rendering -----------------------------------------------------
    def _set_state(self, state: str, status: str, tooltip: Optional[str] = None) -> None:
        """state ∈ {idle, running, done, error}; drives the chunk color via QSS."""
        # Any active state re-enables a previously disabled bar (see set_disabled).
        if not self.isEnabled():
            self.setEnabled(True)
        if self.bar.property("state") != state:
            self.bar.setProperty("state", state)
            self.bar.style().unpolish(self.bar)
            self.bar.style().polish(self.bar)
        color = {"error": C_ANOM, "done": C_ACCENT, "running": C_TEXT}.get(state, C_MUTED)
        self.status_label.setStyleSheet(f"color: {color}; font-size: 10px;")
        self.status_label.setText(status)
        base = f"{self._name}: {tooltip or status}"
        hint = self._click_hint(state)
        self.setToolTip(f"{base}\n({hint})" if hint else base)

    def _click_hint(self, state: str) -> Optional[str]:
        """What a click will do in ``state`` — surfaced in the tooltip so it's discoverable."""
        if self._click_handler is None:
            return None
        if state == "running":
            return "click to pause"
        if state == "done":
            return None  # complete: clicking is a no-op
        if state == "error":
            return "click to retry"
        return "click to resume"

    def set_click_handler(self, fn: Optional[Callable[[], None]]) -> None:
        """Set the handler the app runs when this bar is clicked (it decides pause/resume)."""
        self._click_handler = fn

    def set_running(self, done: int, total: int, status: Optional[str] = None) -> None:
        """Render a live/running task at ``done/total`` (indeterminate when ``total`` is 0)."""
        if total > 0:
            self.bar.setRange(0, total)
            self.bar.setValue(min(done, total))
        else:
            self.bar.setRange(0, 0)
        self._set_state("running", status if status is not None else f"{done}/{total}")

    def mousePressEvent(self, event) -> None:
        if self._click_handler is not None and self.isEnabled():
            self._click_handler()
        super().mousePressEvent(event)

    # --- worker wiring -------------------------------------------------------
    def start(self, worker: FunctionWorker, on_success=None, on_failed=None) -> None:
        self.cancel()
        self.worker = worker
        self.bar.setRange(0, 0)  # busy/indeterminate until the first progress update
        self._set_state("running", "starting…")

        def succeeded(result):
            self._finish_done()
            if on_success is not None:
                on_success(result)

        def failed(error):
            self._finish_error(error)
            if on_failed is not None:
                on_failed(error)

        worker.progress.connect(self._on_progress)
        worker.message.connect(self._on_message)
        worker.succeeded.connect(succeeded)
        worker.failed.connect(failed)
        worker.start()

    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.bar.setRange(0, total)
            self.bar.setValue(done)
            self._set_state("running", f"{done}/{total}")

    def _on_message(self, text: str) -> None:
        self._set_state("running", text, tooltip=text)

    def _finish_done(self) -> None:
        self.bar.setRange(0, 100)
        self.bar.setValue(100)
        self._set_state("done", "done")
        self.worker = None

    def _finish_error(self, error: str) -> None:
        self.bar.setRange(0, 100)
        self.bar.setValue(100)
        self._set_state("error", "error", tooltip=error)
        self.worker = None

    def cancel(self) -> None:
        if self.worker is not None:
            try:
                self.worker.cancel()
            except RuntimeError:
                pass
            for signal in (self.worker.progress, self.worker.message, self.worker.succeeded, self.worker.failed):
                try:
                    signal.disconnect()
                except (TypeError, RuntimeError):
                    pass
            self.worker = None

    def set_idle(self, text: str = "idle") -> None:
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self._set_state("idle", text)

    def set_done(self, text: str = "done", tooltip: Optional[str] = None) -> None:
        self.bar.setRange(0, 100)
        self.bar.setValue(100)
        self._set_state("done", text, tooltip=tooltip)

    def set_error(self, text: str, tooltip: Optional[str] = None) -> None:
        self.bar.setRange(0, 100)
        self.bar.setValue(100)
        self._set_state("error", text, tooltip=tooltip)

    def set_disabled(self, text: str, tooltip: Optional[str] = None) -> None:
        """Gray the bar out and make it non-interactive (e.g. nothing to do).

        Used for the whole-dataset prefetch when only one embryo is loaded. Any later
        ``start``/``set_*`` call re-enables the widget (they route through ``_set_state``).
        """
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self._set_state("idle", text, tooltip=tooltip)
        self.setEnabled(False)


class PiPContainer(QWidget):
    """Holds a full-bleed main widget with a small inset widget (picture-in-picture)
    floated over its bottom-right corner. The inset is repositioned on resize."""

    def __init__(self, main_widget: QWidget, inset_widget: QWidget, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(main_widget)
        self._inset = inset_widget
        inset_widget.setParent(self)
        inset_widget.raise_()
        self._margin = 14

    def set_inset_visible(self, visible: bool) -> None:
        self._inset.setVisible(visible)
        if visible:
            self._reposition()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition()

    def _reposition(self) -> None:
        side = max(150, min(260, int(min(self.width(), self.height()) * 0.32)))
        self._inset.setFixedSize(side, side)
        self._inset.move(self.width() - side - self._margin, self.height() - side - self._margin)
        self._inset.raise_()


class FunctionWorker(QThread):
    """Runs a backend callable off the UI thread so labeling never blocks.

    The callable is invoked as ``fn(progress_cb, message_cb, should_cancel)``:
    ``progress_cb(done, total)`` and ``message_cb(text)`` post updates back to the UI,
    and ``should_cancel()`` returns True once the user cancels. The result (or error
    text) is delivered on the UI thread via the signals below.
    """

    progress = pyqtSignal(int, int)
    message = pyqtSignal(str)
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def _should_cancel(self) -> bool:
        return self._cancelled

    def run(self) -> None:  # executed on the worker thread
        try:
            result = self._fn(
                lambda done, total: self.progress.emit(int(done), int(total)),
                lambda text: self.message.emit(str(text)),
                self._should_cancel,
            )
            self.succeeded.emit(result)
        except Exception as exc:  # surfaced to the user via the failed signal
            self.failed.emit(str(exc))


class SetupModelsDialog(QDialog):
    """Model setup: pick a local model, download one from S3, or fetch the RCNN detector.

    Replaces the old single combo box that crammed three unrelated operations into one
    list. Here *selection* (the local-model list + "Use selected model") is separated from
    *actions* (the S3 / RCNN download buttons), and a status block up top shows what's
    active, whether the RCNN detector is installed, and the AWS sign-in state. The heavy
    work runs through the parent app's existing worker helpers; this dialog passes
    ``refresh_status`` as their completion hook so it updates in place when they finish.
    """

    def __init__(self, app: "EmbryoLabelingApp"):
        super().__init__(app)
        self.app = app
        self.setWindowTitle("Setup Models")
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self.status_label = QLabel()
        self.status_label.setTextFormat(Qt.RichText)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addWidget(self._hsep())

        local_title = QLabel("Local models")
        local_title.setStyleSheet("font-weight: 600;")
        layout.addWidget(local_title)

        self.model_list = QListWidget()
        self.model_list.itemDoubleClicked.connect(lambda _item: self._use_selected())
        layout.addWidget(self.model_list)

        self.use_button = QPushButton("Use selected model")
        self.use_button.clicked.connect(self._use_selected)
        layout.addWidget(self.use_button)

        layout.addWidget(self._hsep())

        downloads_title = QLabel("Download from S3")
        downloads_title.setStyleSheet("font-weight: 600;")
        layout.addWidget(downloads_title)

        actions = QHBoxLayout()
        self.s3_button = QPushButton("Download model…")
        self.s3_button.clicked.connect(self._download_from_s3)
        self.rcnn_button = QPushButton("Fetch RCNN detector")
        self.rcnn_button.clicked.connect(self._fetch_rcnn)
        actions.addWidget(self.s3_button)
        actions.addWidget(self.rcnn_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.refresh_status()

    @staticmethod
    def _hsep() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {C_BORDER};")
        return line

    def refresh_status(self) -> None:
        """Re-read model/RCNN/AWS state and rebuild the status block + model list."""
        self.app.refresh_aws_state()
        active = self.app.selected_model
        active_ready = bool(active) and not setup_models.missing_files(active)
        rcnn_ok = setup_models.rcnn_present()
        aws_ok = self.app._aws_authenticated

        self.status_label.setText(
            f"<b>Active model:</b> {active or '—'} "
            f"({'ready' if active_ready else 'weights not downloaded'})<br>"
            f"<b>RCNN detector:</b> rcnn.pt "
            f"({'installed' if rcnn_ok else 'missing — predictions &amp; ROI need this'})<br>"
            f"<b>AWS:</b> {self.app._aws_status_msg}"
        )

        self.model_list.clear()
        local = setup_models.local_models()
        for name in local:
            item = QListWidgetItem(f"{name}   (active)" if name == active else name)
            item.setData(Qt.UserRole, name)
            self.model_list.addItem(item)
            if name == active:
                self.model_list.setCurrentItem(item)
        if not local:
            placeholder = QListWidgetItem("No local models — use “Download model…”.")
            placeholder.setFlags(Qt.NoItemFlags)
            self.model_list.addItem(placeholder)
        if self.model_list.currentItem() is None and local:
            self.model_list.setCurrentRow(0)

        self.use_button.setEnabled(bool(local))
        self.rcnn_button.setText("Re-fetch RCNN detector" if rcnn_ok else "Fetch RCNN detector")
        # S3 buttons stay enabled even when signed out so clicking surfaces the
        # sign-in guidance; the tooltip explains the state up front.
        aws_tip = "" if aws_ok else self.app._aws_status_msg
        self.s3_button.setToolTip(aws_tip)
        self.rcnn_button.setToolTip(aws_tip)

    def _selected_model_name(self) -> Optional[str]:
        item = self.model_list.currentItem()
        return item.data(Qt.UserRole) if item is not None else None

    def _use_selected(self) -> None:
        name = self._selected_model_name()
        if name:
            self.app._select_model(name, on_done=self.refresh_status)

    def _download_from_s3(self) -> None:
        self.app._fetch_and_choose_s3_model(on_done=self.refresh_status)

    def _fetch_rcnn(self) -> None:
        self.app._fetch_rcnn(on_done=self.refresh_status)


class EmbryoLabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset: Optional[DataSet] = None
        self.current_patient_ts = None
        self.current_timepoint = 0
        self.current_depth = 0
        self.all_depths_window: Optional[AllDepthsWindow] = None
        self.dashboard_window: Optional[DashboardWindow] = None
        self.timeline_window: Optional["TimelineWindow"] = None
        self.segmentation_window: Optional["SegmentationWindow"] = None

        # Assisted-labeling state (predictions / OCR / RCNN / anomalies).
        self.selected_model: Optional[str] = DEFAULT_MODEL
        self.model_is_default = True
        self._aws_authenticated = False
        self._aws_status_msg = "AWS sign-in not checked yet."
        self.show_postprocessed = True
        self._pred_arrays: Optional[Dict] = None
        self._ocr_arrays: Optional[Dict] = None
        self._anomaly_cache: Optional[Dict] = None
        self._stage_start: Optional[int] = None
        self._stage_end: Optional[int] = None

        # Per-embryo navigation state (timepoint + focal depth), keyed by patient
        # directory, so toggling between loaded embryos returns each to where it was
        # left instead of snapping back to t0 / the default depth.
        self._patient_view_state: Dict[str, Dict[str, int]] = {}

        # Background-work bookkeeping (workers kept alive until they finish).
        self._workers: List[FunctionWorker] = []
        # Append-only: keeps background-task QThreads referenced for their lifetime so
        # they aren't GC'd mid-run (which would warn / crash) even after a patient swap.
        self._task_workers: List[FunctionWorker] = []
        self._progress_dialog: Optional[QProgressDialog] = None
        self._roi_worker: Optional[FunctionWorker] = None
        # Background sweep: every loaded embryo's tasks auto-start as their own workers,
        # but only a bounded number of *non-current* embryos compute at once (the model's
        # forward pass is globally serialized anyway) so the thread count stays sane. The
        # current embryo is always allowed to run (labeling-speed priority). The user can
        # pause the whole background sweep from the "All embryos" bar (sticky).
        self._bg_paused_all = False

        # Per-(embryo, task) background jobs. A task belongs to its embryo and runs to
        # completion regardless of which embryo is displayed — the OCR/Pred/ROI bars are
        # just *views* of the current embryo's tasks. ``_bg_workers`` holds live workers,
        # ``_bg_progress`` the last (done, total) per key (so a bar restores on return),
        # and ``_paused`` the keys the user has explicitly paused (sticky across nav).
        self._bg_workers: Dict[Tuple[str, str], FunctionWorker] = {}
        self._bg_progress: Dict[Tuple[str, str], Tuple[int, int]] = {}
        self._paused_tasks: Set[Tuple[str, str]] = set()
        self._pending_roi = None
        self._roi_detect_failed = False

        self.initUI()

    def initUI(self) -> None:
        self.setWindowTitle("Embryo Labeling App")
        self.setGeometry(100, 100, 1600, 940)
        self.setStyleSheet(STYLESHEET)
        self._build_menu()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(8, 8, 8, 8)
        body.setSpacing(8)
        root.addLayout(body, 1)

        body.addLayout(self._build_image_column(), 1)
        body.addWidget(self._build_rail())

        self.setAcceptDrops(True)
        self._build_shortcuts()
        self.toggle_roi()

        # Initialize the model status chip + AWS-gated actions from current state.
        self.refresh_aws_state()

    # ------------------------------------------------------------------
    # Cockpit builders
    # ------------------------------------------------------------------
    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(46)
        row = QHBoxLayout(header)
        row.setContentsMargins(10, 4, 10, 4)
        row.setSpacing(10)

        self.patient_combo = QComboBox()
        self.patient_combo.setMinimumWidth(200)
        self.patient_combo.setToolTip("Patient time series")
        self.patient_combo.currentIndexChanged.connect(self.patient_selection_changed)
        row.addWidget(self.patient_combo)

        row.addWidget(self._vsep())

        # Depth nav: ◂ [F0] ▸ (Up/Down arrows also move depth).
        self.depth_prev_btn = QToolButton()
        self.depth_prev_btn.setText("◂")
        self.depth_prev_btn.setToolTip("Lower focal depth (↓)")
        self.depth_prev_btn.clicked.connect(self.show_previous_depth)
        self.depth_indicator = QLabel("—")
        self.depth_indicator.setAlignment(Qt.AlignCenter)
        self.depth_indicator.setMinimumWidth(48)
        self.depth_indicator.setStyleSheet("font-weight: 600;")
        self.depth_next_btn = QToolButton()
        self.depth_next_btn.setText("▸")
        self.depth_next_btn.setToolTip("Higher focal depth (↑)")
        self.depth_next_btn.clicked.connect(self.show_next_depth)
        row.addWidget(self.depth_prev_btn)
        row.addWidget(self.depth_indicator)
        row.addWidget(self.depth_next_btn)

        self.timepoint_indicator = QLabel("t —")
        self.timepoint_indicator.setStyleSheet(f"color: {C_MUTED};")
        self.timepoint_indicator.setToolTip("Timepoint (← → to move)")
        row.addWidget(self.timepoint_indicator)

        row.addStretch(1)

        # View toggles. Kept as QCheckBoxes so the rest of the app drives them by state.
        self.show_roi_checkbox = QCheckBox("ROI")
        self.show_roi_checkbox.setChecked(True)
        self.show_roi_checkbox.setToolTip("Show the ROI inset")
        self.show_roi_checkbox.stateChanged.connect(self.toggle_roi)
        self.show_all_depths_checkbox = QCheckBox("All depths")
        self.show_all_depths_checkbox.setToolTip("Open a window showing every focal depth at once")
        self.show_all_depths_checkbox.stateChanged.connect(self.toggle_all_depths)
        for box in (self.show_roi_checkbox, self.show_all_depths_checkbox):
            row.addWidget(box)

        self.segment_btn = QToolButton()
        self.segment_btn.setText("✏ Segment")
        self.segment_btn.setToolTip("Open the segmentation pane for tPN / tB masks (Ctrl+G)")
        self.segment_btn.clicked.connect(self.open_segmentation_window)
        row.addWidget(self.segment_btn)

        row.addWidget(self._vsep())

        # Persistent model status chip: shows the active model + a non-default badge,
        # and the AWS sign-in state. Click to open Setup Models.
        self.model_chip = QToolButton()
        self.model_chip.setAutoRaise(True)
        self.model_chip.setCursor(Qt.PointingHandCursor)
        self.model_chip.clicked.connect(self.setup_models_dialog)
        row.addWidget(self.model_chip)

        return header

    def _build_image_column(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(8)

        self.image_viewer = pg.ImageView(view=pg.PlotItem())
        configure_image_view(self.image_viewer)

        # ROI inset (picture-in-picture) floated over the main image's corner.
        self.roi_frame = QFrame()
        self.roi_frame.setObjectName("PiP")
        roi_layout = QVBoxLayout(self.roi_frame)
        roi_layout.setContentsMargins(2, 2, 2, 2)
        self.roi_viewer = pg.ImageView(view=pg.PlotItem())
        configure_image_view(self.roi_viewer)
        roi_layout.addWidget(self.roi_viewer)

        self.pip = PiPContainer(self.image_viewer, self.roi_frame)
        column.addWidget(self.pip, 1)

        column.addWidget(self._build_hud())
        column.addWidget(self._build_timeline_bar())
        return column

    def _build_hud(self) -> QWidget:
        hud = QFrame()
        hud.setObjectName("Hud")
        layout = QVBoxLayout(hud)
        layout.setContentsMargins(12, 7, 12, 7)
        layout.setSpacing(2)

        self.hud_label = QLabel("Load a dataset to begin (File ▸ Open, or drag a folder in)")
        self.hud_label.setTextFormat(Qt.RichText)
        self.hud_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(self.hud_label)

        self.info_label_bottom = QLabel("")
        self.info_label_bottom.setStyleSheet(f"color: {C_MUTED}; font-size: 10px;")
        layout.addWidget(self.info_label_bottom)
        return hud

    def _build_timeline_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("TimelineBar")
        layout = QVBoxLayout(bar)
        layout.setContentsMargins(10, 6, 10, 8)
        layout.setSpacing(4)

        head = QHBoxLayout()
        title = QLabel("TIMELINE")
        title.setObjectName("RailTitle")
        head.addWidget(title)
        head.addStretch(1)

        self.seq_toggle_btn = QToolButton()
        self.seq_toggle_btn.setText("postprocessed")
        self.seq_toggle_btn.setCheckable(True)
        self.seq_toggle_btn.setChecked(True)
        self.seq_toggle_btn.setToolTip("Toggle postprocessed / raw-argmax prediction sequence")
        self.seq_toggle_btn.clicked.connect(self.toggle_prediction_sequence)
        head.addWidget(self.seq_toggle_btn)

        self.expand_timeline_btn = QToolButton()
        self.expand_timeline_btn.setText("⤢")
        self.expand_timeline_btn.setToolTip("Open the full timeline window")
        self.expand_timeline_btn.clicked.connect(self.open_timeline)
        head.addWidget(self.expand_timeline_btn)
        layout.addLayout(head)

        self.timeline_strip = TimelineView(self, compact=True)
        self.timeline_strip.setMinimumHeight(118)
        self.timeline_strip.setMaximumHeight(150)
        layout.addWidget(self.timeline_strip)
        return bar

    def _build_rail(self) -> QWidget:
        rail = QFrame()
        rail.setObjectName("Rail")
        rail.setFixedWidth(232)
        rail.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        labels_title = QLabel("LABELS")
        labels_title.setObjectName("RailTitle")
        layout.addWidget(labels_title)

        # Label chips: 2-column grid of checkable buttons with their hotkey shown.
        self.label_buttons = QButtonGroup(self)
        self.label_class_by_button: Dict[QPushButton, str] = {}
        self.label_chip_by_class: Dict[str, QPushButton] = {}
        label_grid = QGridLayout()
        label_grid.setSpacing(5)
        for index, cls in enumerate(CLASSES):
            chip = QPushButton(f"{LABEL_HOTKEYS[cls]}   {cls}")
            chip.setCheckable(True)
            chip.setProperty("chip", "label")
            chip.setProperty("predicted", "false")
            chip.setCursor(Qt.PointingHandCursor)
            chip.clicked.connect(lambda _checked, c=cls: self.select_label_by_class(c))
            self.label_buttons.addButton(chip)
            self.label_class_by_button[chip] = cls
            self.label_chip_by_class[cls] = chip
            label_grid.addWidget(chip, index // 2, index % 2)
        layout.addLayout(label_grid)

        layout.addStretch(1)

        # Background-task progress bars for the current embryo (OCR / predictions / ROI
        # boxes). Auto-started on display and kept running across navigation; click a
        # running bar to pause it, a paused one to resume. Tasks belong to their embryo.
        tasks_title = QLabel("BACKGROUND TASKS")
        tasks_title.setObjectName("RailTitle")
        layout.addWidget(tasks_title)
        self.task_ocr = TaskProgressBar("OCR")
        self.task_pred = TaskProgressBar("Pred")
        self.task_bbox = TaskProgressBar("ROI")
        # Whole-dataset prefetch: computes OCR/predictions/ROI for every *other* loaded
        # embryo in the background so results are ready before the user navigates to them.
        self.task_prefetch = TaskProgressBar("All embryos")
        for task_bar in (self.task_ocr, self.task_pred, self.task_bbox, self.task_prefetch):
            layout.addWidget(task_bar)
        # Each bar is a view of the *current* embryo's task. Clicking toggles pause/resume
        # for that (embryo, task) only — running tasks keep running across navigation, and a
        # paused task stays paused until the user resumes it here. Done bars are no-ops.
        self.task_ocr.set_click_handler(lambda: self._on_task_bar_clicked("ocr"))
        self.task_pred.set_click_handler(lambda: self._on_task_bar_clicked("pred"))
        self.task_bbox.set_click_handler(lambda: self._on_task_bar_clicked("bbox"))
        self.task_prefetch.set_click_handler(self._on_prefetch_bar_clicked)
        # The dataset-wide sweep only applies to multi-embryo datasets; gray it out until
        # more than one embryo is loaded.
        self.task_prefetch.set_disabled(
            "single embryo",
            tooltip="The dataset-wide sweep runs only when more than one embryo is loaded.",
        )

        layout.addSpacing(6)
        hint = QLabel(
            "← → timepoint   ↑ ↓ depth\n"
            "A accept pred\n"
            "[ ] mark stage · F fill\n"
            "N / P next/prev anomaly"
        )
        hint.setStyleSheet(f"color: {C_MUTED}; font-size: 10px;")
        layout.addWidget(hint)
        return rail

    def _vsep(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet(f"color: {C_BORDER};")
        return line

    def _build_shortcuts(self) -> None:
        self.shortcut_up = QShortcut(QtCore.Qt.Key_Up, self)
        self.shortcut_up.activated.connect(self.show_next_depth)
        self.shortcut_down = QShortcut(QtCore.Qt.Key_Down, self)
        self.shortcut_down.activated.connect(self.show_previous_depth)
        self.shortcut_left = QShortcut(QtCore.Qt.Key_Left, self)
        self.shortcut_left.activated.connect(self.show_previous_timepoint)
        self.shortcut_right = QShortcut(QtCore.Qt.Key_Right, self)
        self.shortcut_right.activated.connect(self.show_next_timepoint)

        # Single-key class hotkeys (0-8 + QWERT).
        self._label_shortcuts = []
        for cls, key in LABEL_HOTKEYS.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda c=cls: self.select_label_by_class(c))
            self._label_shortcuts.append(shortcut)

        # Assisted-labeling shortcuts (keyboard-first stage auto-fill + anomaly review).
        self.shortcut_stage_start = QShortcut(QtCore.Qt.Key_BracketLeft, self)
        self.shortcut_stage_start.activated.connect(self.mark_stage_start)
        self.shortcut_stage_end = QShortcut(QtCore.Qt.Key_BracketRight, self)
        self.shortcut_stage_end.activated.connect(self.mark_stage_end)
        self.shortcut_fill_stage = QShortcut(QtCore.Qt.Key_F, self)
        self.shortcut_fill_stage.activated.connect(self.fill_stage_with_selected)
        self.shortcut_accept_pred = QShortcut(QtCore.Qt.Key_A, self)
        self.shortcut_accept_pred.activated.connect(self.accept_prediction_current)
        self.shortcut_next_anomaly = QShortcut(QtCore.Qt.Key_N, self)
        self.shortcut_next_anomaly.activated.connect(self.goto_next_anomaly)
        self.shortcut_prev_anomaly = QShortcut(QtCore.Qt.Key_P, self)
        self.shortcut_prev_anomaly.activated.connect(self.goto_prev_anomaly)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_action = QAction("Open Dataset…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_dataset_dialog)
        file_menu.addAction(open_action)

        tools_menu = self.menuBar().addMenu("Tools")
        # Keep AWS-dependent actions (Re-run Predictions) in sync with sign-in state.
        tools_menu.aboutToShow.connect(self.refresh_aws_state)
        dashboard_action = QAction("Dashboard", self)
        dashboard_action.triggered.connect(self.open_dashboard)
        tools_menu.addAction(dashboard_action)

        timeline_action = QAction("Timeline", self)
        timeline_action.triggered.connect(self.open_timeline)
        tools_menu.addAction(timeline_action)

        segmentation_action = QAction("Segmentation Pane", self)
        segmentation_action.setShortcut("Ctrl+G")
        segmentation_action.triggered.connect(self.open_segmentation_window)
        tools_menu.addAction(segmentation_action)

        tools_menu.addSeparator()
        setup_action = QAction("Setup Models…", self)
        # On macOS, Qt's native menu bar applies a text heuristic that relocates any item
        # containing "setup"/"config"/"options"/"settings"/"preferences" into the
        # application menu (as Preferences) — which is why "Setup Models…" vanished from
        # Tools. Pin NoRole so it stays where it's added.
        setup_action.setMenuRole(QAction.NoRole)
        setup_action.triggered.connect(self.setup_models_dialog)
        tools_menu.addAction(setup_action)

        tools_menu.addSeparator()
        ocr_action = QAction("Run OCR Times", self)
        ocr_action.triggered.connect(self.run_ocr)
        tools_menu.addAction(ocr_action)

        predictions_action = QAction("Run Predictions", self)
        predictions_action.triggered.connect(lambda: self.run_predictions(force=False))
        tools_menu.addAction(predictions_action)

        self.rerun_action = QAction("Re-run Predictions…", self)
        self.rerun_action.triggered.connect(self.rerun_predictions)
        tools_menu.addAction(self.rerun_action)

        detect_rois_action = QAction("Detect ROIs (RCNN)", self)
        detect_rois_action.triggered.connect(self.detect_rois)
        tools_menu.addAction(detect_rois_action)

        # Keyboard-first assisted labeling.
        assist_menu = self.menuBar().addMenu("Assist")
        toggle_seq_action = QAction("Toggle raw / postprocessed", self)
        toggle_seq_action.triggered.connect(self.toggle_prediction_sequence)
        assist_menu.addAction(toggle_seq_action)
        assist_menu.addSeparator()
        mark_start_action = QAction("Mark stage start  [", self)
        mark_start_action.triggered.connect(self.mark_stage_start)
        assist_menu.addAction(mark_start_action)
        mark_end_action = QAction("Mark stage end  ]", self)
        mark_end_action.triggered.connect(self.mark_stage_end)
        assist_menu.addAction(mark_end_action)
        fill_action = QAction("Fill stage with selected class  (F)", self)
        fill_action.triggered.connect(self.fill_stage_with_selected)
        assist_menu.addAction(fill_action)
        accept_stage_action = QAction("Accept predictions for stage", self)
        accept_stage_action.triggered.connect(self.accept_predictions_for_stage)
        assist_menu.addAction(accept_stage_action)
        accept_current_action = QAction("Accept prediction (current)  (A)", self)
        accept_current_action.triggered.connect(self.accept_prediction_current)
        assist_menu.addAction(accept_current_action)
        assist_menu.addSeparator()
        next_anom_action = QAction("Next anomaly  (N)", self)
        next_anom_action.triggered.connect(self.goto_next_anomaly)
        assist_menu.addAction(next_anom_action)
        prev_anom_action = QAction("Previous anomaly  (P)", self)
        prev_anom_action.triggered.connect(self.goto_prev_anomaly)
        assist_menu.addAction(prev_anom_action)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        # Accept every dropped directory and *add* them to the current dataset (drops from
        # different folders accumulate instead of replacing). All embryos stay in the
        # dropdown; File ▸ Open Dataset is the explicit "replace" path.
        directories = [
            url.toLocalFile()
            for url in event.mimeData().urls()
            if os.path.isdir(url.toLocalFile())
        ]
        if directories:
            self.add_directories(directories)

    def open_dataset_dialog(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Open dataset folder")
        if directory:
            self.load_dataset(directory)

    def load_dataset(self, directory: str) -> None:
        """Replace the current dataset with the contents of ``directory`` (menu open)."""
        try:
            self._bg_paused_all = False  # fresh dataset: allow the background sweep again
            self._patient_view_state.clear()  # replacing the dataset: drop stale view state
            self._cancel_all_bg_tasks()  # stop the old dataset's per-embryo tasks
            self.dataset = DataSet(directory)
            self.update_patient_list()
            self.ensure_model_ready()
            self.auto_select_first_patient()
            self._schedule_background_work()
            QMessageBox.information(self, "Success", f"Loaded dataset from {directory}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load dataset from {directory}\n{exc}")

    def add_directories(self, directories: List[str]) -> None:
        """Add embryo/patient directories to the current dataset (drag-and-drop path).

        Bootstraps a new dataset from the first directory if none is loaded yet, then
        merges the rest; otherwise appends to the existing dataset. Duplicates are skipped
        and the user's current patient selection is preserved across the rebuild.
        """
        try:
            # New embryos added — resume the sweep so it covers them. Existing embryos'
            # tasks keep running (we don't cancel them).
            self._bg_paused_all = False

            bootstrap = self.dataset is None
            pending = list(directories)
            if bootstrap:
                self.dataset = DataSet(pending.pop(0))
            before = self.dataset.num_patients()
            for directory in pending:
                self.dataset.add_directory(directory)
            added = self.dataset.num_patients() - (0 if bootstrap else before)

            self.ensure_model_ready()
            self.update_patient_list()
            if bootstrap:
                self.auto_select_first_patient()
            self._schedule_background_work()

            total = self.dataset.num_patients()
            if added <= 0:
                QMessageBox.information(
                    self, "Already loaded", "Those embryos are already in the dropdown."
                )
            else:
                QMessageBox.information(
                    self, "Embryos added", f"Added {added} embryo(s). {total} now loaded."
                )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to add embryos\n{exc}")

    def _patient_display_names(self) -> List[str]:
        """Dropdown labels for each patient, disambiguated when basenames collide.

        Embryos dropped from different folders often share a basename (e.g. ``Emb1``);
        such duplicates get their parent folder appended so the dropdown stays unambiguous.
        """
        series = self.dataset.get_patient_series() if self.dataset else []
        names = [os.path.basename(patient.directory) for patient in series]
        counts = Counter(names)
        labels: List[str] = []
        for patient, name in zip(series, names):
            if counts[name] > 1:
                parent = os.path.basename(os.path.dirname(patient.directory))
                labels.append(f"{name}  ({parent})")
            else:
                labels.append(name)
        return labels

    def update_patient_list(self) -> None:
        # Preserve the current selection across rebuilds — adding embryos must not yank
        # the user off the patient they're labeling.
        current_dir = self.current_patient_ts.directory if self.current_patient_ts else None
        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()
        restore_index = -1
        if self.dataset:
            series = self.dataset.get_patient_series()
            for index, (patient_ts, label) in enumerate(zip(series, self._patient_display_names())):
                self.patient_combo.addItem(label)
                if patient_ts.directory == current_dir:
                    restore_index = index
        if restore_index >= 0:
            self.patient_combo.setCurrentIndex(restore_index)
        self.patient_combo.blockSignals(False)

    def auto_select_first_patient(self) -> None:
        if self.patient_combo.count() > 0:
            # setCurrentIndex(0) won't emit if already at 0, so trigger explicitly.
            if self.patient_combo.currentIndex() == 0:
                self.patient_selection_changed()
            else:
                self.patient_combo.setCurrentIndex(0)

    def patient_selection_changed(self, _index: int = -1) -> None:
        if not self.dataset:
            return
        index = self.patient_combo.currentIndex()
        series = self.dataset.get_patient_series()
        if not (0 <= index < len(series)):
            return

        self.close_all_depths_window()

        # Remember where we were on the embryo we're leaving so returning to it restores
        # that timepoint/depth instead of snapping back to t0.
        self._save_current_view_state()

        self.current_patient_ts = series[index]
        self._restore_view_state(self.current_patient_ts)

        # Reset per-patient assisted-labeling state.
        self._pred_arrays = None
        self._ocr_arrays = None
        self._anomaly_cache = None
        self._stage_start = None
        self._stage_end = None
        self._pending_roi = None

        # Bind the timeline to the new patient first (it resets to t0), then display the
        # image — display_image moves the timeline marker to the restored timepoint.
        for widget in self._timeline_widgets():
            widget.set_patient(self.current_patient_ts)

        if self.segmentation_window is not None:
            self.segmentation_window.set_patient(self.current_patient_ts)

        self.update_nav_indicators()
        self.display_image()

        # Bind the bars to this embryo's task state, then pump the background sweep: the
        # current embryo's tasks (re)start with priority and a bounded set of other embryos
        # keep computing. Navigation never cancels a running task — tasks belonging to the
        # embryo we just left keep running in the background.
        self.refresh_task_bars(self.current_patient_ts)
        self._schedule_background_work()

    def _save_current_view_state(self) -> None:
        """Stash the current embryo's timepoint + depth so a later return restores it."""
        if self.current_patient_ts is not None:
            self._patient_view_state[self.current_patient_ts.directory] = {
                "timepoint": self.current_timepoint,
                "depth": self.current_depth,
            }

    def _restore_view_state(self, patient_ts) -> None:
        """Set timepoint/depth to where ``patient_ts`` was last left (else its defaults).

        Saved values are clamped to the patient's current ranges so an image set that
        changed since the last visit can't push the view out of bounds.
        """
        saved = self._patient_view_state.get(patient_ts.directory)
        if saved is None:
            self.current_timepoint = 0
            self.current_depth = patient_ts.default_depth_index()
            return
        num_timepoints = patient_ts.num_timepoints()
        self.current_timepoint = min(max(0, saved["timepoint"]), max(0, num_timepoints - 1))
        self.current_depth = min(max(0, saved["depth"]), patient_ts.num_depths() - 1)

    # ------------------------------------------------------------------
    # Image display and navigation
    # ------------------------------------------------------------------
    def display_image(self) -> None:
        if self.current_patient_ts is None:
            return
        try:
            # Some patients only populate a subset of the 7 focal depths (e.g. the
            # 3-depth inference sets). Render an empty depth gracefully instead of
            # raising — timepoint-based controls stay usable.
            if not self.current_patient_ts.depth_has_images(self.current_depth):
                self.image_viewer.clear()
                if self.show_roi_checkbox.isChecked():
                    self.roi_viewer.clear()
                self.update_nav_indicators()
                self.update_label_buttons()
                depth_name = self.current_patient_ts.get_depth_name(self.current_depth)
                self.info_label_bottom.setText(f"No images for depth {depth_name}")
                self.update_prediction_label()
                for widget in self._timeline_widgets():
                    widget.set_current_timepoint(self.current_timepoint)
                return

            image = self.current_patient_ts.get_image(self.current_timepoint, self.current_depth)
            self.image_viewer.setImage(image.T)

            if self.show_roi_checkbox.isChecked():
                roi_image = self.get_ROI(image)
                self.roi_viewer.setImage(roi_image.T)

            self.update_nav_indicators()
            self.update_label_buttons()
            self.update_info_label()
            self.update_prediction_label()

            if self.all_depths_window is not None:
                self.update_all_depths_window()
            for widget in self._timeline_widgets():
                widget.set_current_timepoint(self.current_timepoint)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to display image: {exc}")

    def get_ROI(self, image: np.ndarray) -> np.ndarray:
        """Embryo ROI for the current frame.

        Uses the RCNN box detected at the patient's reference depth (F0 or fallback) for
        this timepoint, reused for whatever focal depth is shown — the embryo sits at the
        same position across focal planes. Falls back to an instant center crop until the
        box exists; detection runs off the UI thread so scrolling is never blocked.
        """
        patient = self.current_patient_ts
        if patient is None:
            return rcnn_roi.center_crop(image)

        ref_depth = rcnn_roi.reference_depth(patient)
        if ref_depth is None:
            return rcnn_roi.center_crop(image)

        roi, is_cached = rcnn_roi.roi_for_display(
            image, patient, ref_depth, self.current_timepoint
        )
        # Skip per-frame detection while a background task that writes this patient's box
        # cache is running (the prediction or the standalone ROI task), to avoid redundant
        # detection of frames it will fill in.
        box_writer_running = (
            (patient.directory, "bbox") in self._bg_workers
            or (patient.directory, "pred") in self._bg_workers
        )
        if (
            not is_cached
            and not self._roi_detect_failed
            and not box_writer_running
        ):
            self._schedule_roi_detection(ref_depth, self.current_timepoint)
        return roi

    # ------------------------------------------------------------------
    # Background RCNN ROI detection for the current frame
    # ------------------------------------------------------------------
    def _schedule_roi_detection(self, depth_name: str, timepoint: int) -> None:
        """Detect+cache the box for one frame off-thread; only the latest is kept.

        Detection runs the same triplet-based logic as the batch / prediction paths (on a
        worker thread) so every producer caches an identical box for this timepoint, reused
        across all focal depths.
        """
        patient = self.current_patient_ts
        if patient is None or depth_name not in patient.depths:
            return
        request = (depth_name, timepoint)
        if self._roi_worker is not None and self._roi_worker.isRunning():
            self._pending_roi = request
            return
        self._start_roi_worker(request)

    def _start_roi_worker(self, request) -> None:
        depth_name, timepoint = request
        patient = self.current_patient_ts
        if patient is None:
            return
        ref_index = patient.depths.index(depth_name)

        def job(progress_cb, message_cb, should_cancel):
            bbox = rcnn_roi.detect_box_for_timepoint(patient, timepoint, ref_index)
            patient.set_rcnn_box(
                depth_name, timepoint, list(bbox) if bbox is not None else None, save=True
            )
            return bbox

        worker = FunctionWorker(job, self)
        self._roi_worker = worker
        worker.succeeded.connect(lambda _result, d=depth_name, t=timepoint: self._on_roi_detected(d, t))
        worker.failed.connect(self._on_roi_detect_failed)
        worker.finished.connect(self._on_roi_worker_finished)
        worker.start()

    def _on_roi_detected(self, depth_name: str, timepoint: int) -> None:
        # Refresh the ROI view if still on the detected timepoint — the box is reused
        # across depths, so the displayed depth doesn't matter.
        if (
            self.current_patient_ts is not None
            and self.current_timepoint == timepoint
            and self.show_roi_checkbox.isChecked()
        ):
            self.refresh_roi_view()

    def _on_roi_detect_failed(self, error: str) -> None:
        # Detection failed (e.g. weights/torch missing): stop auto-detecting and tell
        # the user once. Labeling continues with center crops.
        self._roi_detect_failed = True
        QMessageBox.warning(
            self,
            "ROI detection unavailable",
            f"RCNN ROI detection failed and was disabled:\n{error}\n\n"
            "Use Tools > Setup Models to install weights, then Tools > Detect ROIs "
            "to retry.",
        )

    def _on_roi_worker_finished(self) -> None:
        self._roi_worker = None
        if self._pending_roi is not None and not self._roi_detect_failed:
            pending, self._pending_roi = self._pending_roi, None
            self._start_roi_worker(pending)

    def refresh_roi_view(self) -> None:
        if (
            self.current_patient_ts is None
            or not self.show_roi_checkbox.isChecked()
            or not self.current_patient_ts.depth_has_images(self.current_depth)
        ):
            return
        image = self.current_patient_ts.get_image(self.current_timepoint, self.current_depth)
        self.roi_viewer.setImage(self.get_ROI(image).T)

    def show_previous_depth(self) -> None:
        if self.current_patient_ts is None:
            return
        self.current_depth = self._adjacent_populated_depth(-1)
        self.display_image()

    def show_next_depth(self) -> None:
        if self.current_patient_ts is None:
            return
        self.current_depth = self._adjacent_populated_depth(1)
        self.display_image()

    def _adjacent_populated_depth(self, step: int) -> int:
        """Next/previous depth that has images (skips empty depth slots)."""
        patient = self.current_patient_ts
        index = self.current_depth + step
        while 0 <= index < patient.num_depths():
            if patient.depth_has_images(index):
                return index
            index += step
        return self.current_depth  # no populated depth that way; stay put

    def show_previous_timepoint(self) -> None:
        if self.current_patient_ts is None:
            return
        self.current_timepoint = max(0, self.current_timepoint - 1)
        self.display_image()

    def show_next_timepoint(self) -> None:
        if self.current_patient_ts is None:
            return
        self.current_timepoint = min(self.current_patient_ts.num_timepoints() - 1, self.current_timepoint + 1)
        self.display_image()

    def update_info_label(self) -> None:
        if self.current_patient_ts is None:
            return
        image_path = self.current_patient_ts.image_paths[self.current_depth][self.current_timepoint]
        self.info_label_bottom.setText(os.path.basename(image_path))
        self.info_label_bottom.setToolTip(image_path)
        self.image_viewer.setToolTip(image_path)

    # ------------------------------------------------------------------
    # Header nav indicators (depth name + timepoint counter)
    # ------------------------------------------------------------------
    def update_nav_indicators(self) -> None:
        patient = self.current_patient_ts
        if patient is None:
            self.depth_indicator.setText("—")
            self.timepoint_indicator.setText("t —")
            return
        self.depth_indicator.setText(patient.get_depth_name(self.current_depth))
        self.timepoint_indicator.setText(f"t {self.current_timepoint} / {patient.num_timepoints() - 1}")

    # ------------------------------------------------------------------
    # Label controls
    # ------------------------------------------------------------------
    def _selected_label(self) -> Optional[str]:
        button = self.label_buttons.checkedButton()
        return self.label_class_by_button.get(button) if button is not None else None

    def _set_chip_property(self, chip: QPushButton, name: str, value: bool) -> None:
        """Set a bool QSS property and re-polish so the stylesheet rule re-applies."""
        new = "true" if value else "false"
        if chip.property(name) != new:
            chip.setProperty(name, new)
            chip.style().unpolish(chip)
            chip.style().polish(chip)

    def update_label_buttons(self) -> None:
        if self.current_patient_ts is None:
            return
        label = self.current_patient_ts.get_label(self.current_timepoint)
        self.label_buttons.setExclusive(False)
        for chip, cls in self.label_class_by_button.items():
            chip.setChecked(cls == label)
        self.label_buttons.setExclusive(True)

        # Outline the predicted class so the user can see what 'A' would accept.
        predicted = None
        pred = self._get_pred_arrays()
        if pred and self.current_timepoint < pred["num_timepoints"]:
            key = "post_class" if self.show_postprocessed else "raw_class"
            predicted = pred[key][self.current_timepoint]
        for cls, chip in self.label_chip_by_class.items():
            self._set_chip_property(chip, "predicted", cls == predicted)

    def select_label_by_class(self, cls: str) -> None:
        """Apply ``cls`` to the current timepoint (chip click or keyboard hotkey)."""
        if self.current_patient_ts is None:
            return
        chip = self.label_chip_by_class.get(cls)
        if chip is None:
            return
        self.label_buttons.setExclusive(False)
        for button in self.label_buttons.buttons():
            button.setChecked(button is chip)
        self.label_buttons.setExclusive(True)
        self.current_patient_ts.set_label(self.current_timepoint, cls)
        self._after_label_change()

    # ------------------------------------------------------------------
    # Optional panels/windows
    # ------------------------------------------------------------------
    def toggle_roi(self) -> None:
        self.pip.set_inset_visible(self.show_roi_checkbox.isChecked())
        if (
            self.show_roi_checkbox.isChecked()
            and self.current_patient_ts is not None
            and self.current_patient_ts.depth_has_images(self.current_depth)
        ):
            image = self.current_patient_ts.get_image(self.current_timepoint, self.current_depth)
            self.roi_viewer.setImage(self.get_ROI(image).T)

    def toggle_all_depths(self) -> None:
        if self.show_all_depths_checkbox.isChecked():
            self.open_all_depths_window()
        else:
            self.close_all_depths_window()

    def open_all_depths_window(self) -> None:
        if self.current_patient_ts is None:
            self.show_all_depths_checkbox.blockSignals(True)
            self.show_all_depths_checkbox.setChecked(False)
            self.show_all_depths_checkbox.blockSignals(False)
            QMessageBox.warning(self, "No patient selected", "Load a dataset and select a patient first.")
            return
        if self.all_depths_window is None:
            self.all_depths_window = AllDepthsWindow(
                self,
                ORDERED_FOCAL_DEPTH,
                self.current_patient_ts,
                self.current_timepoint,
            )
            self.all_depths_window.show()

    def close_all_depths_window(self) -> None:
        if self.all_depths_window is not None:
            window = self.all_depths_window
            self.all_depths_window = None
            window.close()
        self.show_all_depths_checkbox.blockSignals(True)
        self.show_all_depths_checkbox.setChecked(False)
        self.show_all_depths_checkbox.blockSignals(False)

    def update_all_depths_window(self) -> None:
        if self.current_patient_ts is not None and self.all_depths_window is not None:
            self.all_depths_window.update_images(self.current_patient_ts, self.current_timepoint)

    def open_dashboard(self) -> None:
        if self.dataset is None:
            QMessageBox.warning(self, "No dataset loaded", "Load a dataset before opening the dashboard.")
            return
        self.dashboard_window = DashboardWindow(self, self.dataset, CLASSES)
        self.dashboard_window.show()

    def open_timeline(self) -> None:
        if self.current_patient_ts is None:
            QMessageBox.warning(self, "No patient selected", "Load a dataset and select a patient first.")
            return
        if self.timeline_window is None:
            self.timeline_window = TimelineWindow(self, self.current_patient_ts)
        else:
            self.timeline_window.set_patient(self.current_patient_ts)
        self.timeline_window.set_current_timepoint(self.current_timepoint)
        self.timeline_window.show()
        self.timeline_window.raise_()

    def open_segmentation_window(self) -> None:
        if self.current_patient_ts is None:
            QMessageBox.warning(self, "No patient selected", "Load a dataset and select a patient first.")
            return
        if self.segmentation_window is None:
            self.segmentation_window = SegmentationWindow(
                self,
                self.current_patient_ts,
                timepoint=self.current_timepoint,
                depth=self.current_depth,
            )
        else:
            self.segmentation_window.set_patient(self.current_patient_ts)
        self.segmentation_window.show()
        self.segmentation_window.raise_()

    # ==================================================================
    # Background workers + progress
    # ==================================================================
    def _run_worker(self, title, job, on_success=None, *, determinate=True, cancelable=True):
        """Run ``job(progress_cb, message_cb, should_cancel)`` off-thread with a dialog."""
        maximum = 100 if determinate else 0
        dialog = QProgressDialog(title, "Cancel" if cancelable else "", 0, maximum, self)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setValue(0)

        worker = FunctionWorker(job, self)
        self._workers.append(worker)

        def on_progress(done, total):
            if total > 0:
                dialog.setMaximum(total)
                dialog.setValue(done)

        def on_message(text):
            dialog.setLabelText(f"{title}\n{text}")

        def cleanup():
            dialog.reset()
            dialog.close()
            if worker in self._workers:
                self._workers.remove(worker)

        def handle_success(result):
            cleanup()
            if on_success is not None:
                on_success(result)

        def handle_failed(error):
            cleanup()
            QMessageBox.critical(self, title, f"{title} failed:\n{error}")

        worker.progress.connect(on_progress)
        worker.message.connect(on_message)
        worker.succeeded.connect(handle_success)
        worker.failed.connect(handle_failed)
        if cancelable:
            dialog.canceled.connect(worker.cancel)
        worker.start()
        dialog.show()
        return worker

    # ==================================================================
    # Model setup (AWS)
    # ==================================================================
    def refresh_aws_state(self) -> None:
        """Re-check AWS sign-in and sync the UI (status chip + AWS-gated actions).

        Cached in ``self._aws_authenticated`` / ``self._aws_status_msg`` so the rest of
        the UI reads one source of truth instead of shelling out repeatedly.
        """
        self._aws_authenticated, self._aws_status_msg = setup_models.auth_status()
        if hasattr(self, "rerun_action"):
            self.rerun_action.setEnabled(self._aws_authenticated)
            self.rerun_action.setToolTip("" if self._aws_authenticated else self._aws_status_msg)
        self._update_model_chip()

    def select_model(self, model_name: str) -> None:
        """Make ``model_name`` the active model and refresh the status chip.

        Single setter for the active model so the default/non-default badge stays
        consistent everywhere it can change.
        """
        self.selected_model = model_name
        self.model_is_default = (model_name == DEFAULT_MODEL)
        self._update_model_chip()

    def _update_model_chip(self) -> None:
        if not hasattr(self, "model_chip"):
            return
        name = self.selected_model
        ready = bool(name) and not setup_models.missing_files(name)
        if self.model_is_default:
            label = "Model: default"
            color = C_ACCENT if ready else C_MUTED
        else:
            short = (name[:16] + "…") if name and len(name) > 17 else (name or "—")
            label = f"Model: {short}  ⚠ non-default"
            color = C_PRED  # orange — draw attention to a non-default model
        if not ready:
            label += "  (not downloaded)"
        self.model_chip.setText(label)
        self.model_chip.setStyleSheet(f"color: {color}; font-size: 11px;")
        state = "ready" if ready else "weights not downloaded"
        self.model_chip.setToolTip(
            f"{name}\nStatus: {state}\nAWS: {self._aws_status_msg}\n\n"
            "Click to choose or download a model."
        )

    def ensure_model_ready(self) -> None:
        """Detect/select a usable model on dataset load (prompts before downloading).

        Default present → use it silently. Default missing but another local model is
        ready → use it and flag it as non-default (via the status chip). Nothing ready →
        offer to download the default if signed in to AWS; otherwise leave the (missing)
        default selected so the chip shows the situation.
        """
        self.refresh_aws_state()

        if not setup_models.missing_files(DEFAULT_MODEL):
            self.select_model(DEFAULT_MODEL)
            return

        ready = [m for m in setup_models.local_models() if not setup_models.missing_files(m)]
        if ready:
            # A non-default model is ready — use it, and the chip flags it as non-default.
            self.select_model(ready[0])
            return

        if self._aws_authenticated:
            response = QMessageBox.question(
                self,
                "Download default model?",
                "No model weights were found locally.\n\nDownload the default model now?\n"
                f"({DEFAULT_MODEL})",
                QMessageBox.Yes | QMessageBox.No,
            )
            if response == QMessageBox.Yes:
                self._download_and_select(DEFAULT_MODEL)
                return

        # Unauthenticated or declined: keep the (missing) default selected; chip shows it.
        self.select_model(DEFAULT_MODEL)

    def setup_models_dialog(self) -> None:
        """Open the model-setup dialog (select a local model / download / fetch RCNN)."""
        SetupModelsDialog(self).exec_()

    def _fetch_rcnn(self, on_done=None) -> None:
        """Download just the RCNN detector weights (repairs a missing/failed rcnn.pt)."""
        if not self._require_aws_or_warn():
            return

        def job(progress_cb, message_cb, should_cancel):
            return setup_models.download_rcnn(
                overwrite=True, progress_cb=message_cb, should_cancel=should_cancel
            )

        def on_success(_downloaded):
            # Re-enable per-frame ROI detection that was disabled when the weights were
            # missing, and refresh the chip + ROI view now that the detector can load.
            self._roi_detect_failed = False
            self._update_model_chip()
            self.refresh_roi_view()
            if on_done is not None:
                on_done()
            QMessageBox.information(
                self, "RCNN ready", "RCNN detector weights (rcnn.pt) are installed."
            )

        self._run_worker("Fetch RCNN weights", job, on_success, determinate=False)

    def _fetch_and_choose_s3_model(self, on_done=None) -> None:
        if not self._require_aws_or_warn():
            return

        def job(progress_cb, message_cb, should_cancel):
            message_cb("Listing models in s3://cfai-model-weights …")
            return setup_models.list_s3_models()

        def on_success(models):
            if not models:
                QMessageBox.information(self, "Setup Models", "No models found in the S3 bucket.")
                return
            choice, ok = QInputDialog.getItem(
                self, "Setup Models", "Choose a model to download/use:", models, 0, False
            )
            if ok and choice:
                self._select_model(choice, on_done=on_done)

        self._run_worker("Fetch S3 model list", job, on_success, determinate=False)

    def _select_model(self, model_name: str, on_done=None) -> None:
        missing = setup_models.missing_files(model_name)
        if not missing:
            self.select_model(model_name)
            if on_done is not None:
                on_done()
            QMessageBox.information(
                self, "Model ready", f"'{model_name}' is ready for predictions."
            )
            return

        if not self._require_aws_or_warn():
            return
        response = QMessageBox.question(
            self,
            "Download weights?",
            f"'{model_name}' is missing {missing}.\nDownload from S3 now?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if response != QMessageBox.Yes:
            return
        self._download_and_select(model_name, on_done=on_done)

    def _download_and_select(self, model_name: str, on_done=None) -> None:
        """Download ``model_name`` (+ rcnn.pt) from S3, then make it the active model."""
        def job(progress_cb, message_cb, should_cancel):
            return setup_models.download_model(
                model_name, progress_cb=message_cb, should_cancel=should_cancel
            )

        def on_success(_downloaded):
            self.select_model(model_name)
            # A fresh model download also pulls rcnn.pt, so ROI detection may be usable again.
            self._roi_detect_failed = False
            if on_done is not None:
                on_done()
            QMessageBox.information(
                self, "Download complete", f"'{model_name}' is ready for predictions."
            )

        self._run_worker(f"Download {model_name}", job, on_success, determinate=False)

    def _require_aws_or_warn(self) -> bool:
        """True if signed in to AWS; otherwise show the guidance message and return False."""
        self.refresh_aws_state()
        if self._aws_authenticated:
            return True
        QMessageBox.information(self, "AWS sign-in required", self._aws_status_msg)
        return False

    def _ensure_model_selected(self) -> Optional[str]:
        """Return a ready-to-use model name, prompting if needed (None if unavailable).

        Offline-friendly: only considers models whose weights are already present, so
        ``Run Predictions`` works without AWS. Use ``rerun_predictions`` to pull a model
        from S3.
        """
        if self.selected_model and not setup_models.missing_files(self.selected_model):
            return self.selected_model
        ready = [m for m in setup_models.local_models() if not setup_models.missing_files(m)]
        if not ready:
            QMessageBox.information(
                self,
                "No model available",
                "No local model has all required weights. Use Tools ▸ Setup Models to "
                "download one (needs AWS access to cfai-model-weights).",
            )
            return None
        if len(ready) == 1:
            self.select_model(ready[0])
            return ready[0]
        choice, ok = QInputDialog.getItem(
            self, "Select Model", "Model to use for predictions:", ready, 0, False
        )
        if not ok:
            return None
        self.select_model(choice)
        return choice

    def rerun_predictions(self) -> None:
        """Re-run predictions with a model chosen from S3 (default first, then bucket).

        AWS-gated: the menu action is disabled when signed out, and this guards again in
        case sign-in lapsed. The chosen model is downloaded first if not already local.
        """
        if self.current_patient_ts is None:
            QMessageBox.warning(self, "No patient selected", "Select a patient first.")
            return
        if not self._require_aws_or_warn():
            return
        patient = self.current_patient_ts

        def job(progress_cb, message_cb, should_cancel):
            message_cb("Listing models in s3://cfai-model-weights …")
            return setup_models.list_s3_models()

        def on_success(s3_models):
            # Default first, then the S3 bucket models (de-duplicated against it).
            items = [DEFAULT_MODEL] + [m for m in s3_models if m != DEFAULT_MODEL]
            labels = [f"{m}  (default)" if m == DEFAULT_MODEL else m for m in items]
            choice, ok = QInputDialog.getItem(
                self,
                "Re-run Predictions",
                "Model to use (downloaded from S3 if not present locally):",
                labels,
                0,
                False,
            )
            if not ok:
                return
            self._run_predictions_with_model(patient, items[labels.index(choice)])

        self._run_worker("Fetch S3 model list", job, on_success, determinate=False)

    def _run_predictions_with_model(self, patient, model_name: str) -> None:
        """Download ``model_name`` if missing, then (re)compute predictions for ``patient``."""
        self.select_model(model_name)
        if setup_models.missing_files(model_name) and not self._require_aws_or_warn():
            return

        def job(progress_cb, message_cb, should_cancel):
            if setup_models.missing_files(model_name):
                setup_models.download_model(
                    model_name, progress_cb=message_cb, should_cancel=should_cancel
                )
            # Deliberate re-run with a chosen model: recompute from scratch rather than
            # resuming whatever a previous model left in the sidecar.
            return predmod.compute_and_save_predictions(
                patient, model_name, progress_cb=progress_cb, should_cancel=should_cancel, resume=False
            )

        self._spawn_bg_worker(patient, "pred", job)

    # ==================================================================
    # Per-(embryo, task) background jobs
    #
    # Each embryo owns its OCR / Pred / ROI tasks. A task runs to completion on its own
    # worker and is NEVER cancelled by navigation — the three bars are views of whichever
    # embryo is displayed. The only way to stop a task is for the user to click its bar
    # (pause, sticky). On display, an embryo's not-done / not-paused / not-running tasks
    # are auto-started; running ones are re-bound to show live progress; paused ones stay
    # paused. The "All embryos" prefetch fills in embryos the user hasn't opened.
    # ==================================================================
    _TASK_BARS = ("ocr", "pred", "bbox")
    # Max number of *non-current* embryos computing at once. The model forward pass is
    # globally serialized, so this mainly bounds the live-thread count; the current embryo
    # is never counted against it.
    _BG_MAX_BACKGROUND = 3
    # Repaint the current embryo's timeline/HUD every this-many computed timepoints (matches
    # the OCR/prediction sidecar flush cadence) so the chart fills in as a task runs.
    _TIMELINE_REFRESH_EVERY = 10

    def _bar_for(self, task: str) -> "TaskProgressBar":
        return {"ocr": self.task_ocr, "pred": self.task_pred, "bbox": self.task_bbox}[task]

    def _is_current_key(self, key: Tuple[str, str]) -> bool:
        return self.current_patient_ts is not None and key[0] == self.current_patient_ts.directory

    def _model_ready(self) -> bool:
        """A model is selected and its weights (incl. rcnn.pt) are present locally."""
        return bool(self.selected_model) and not setup_models.missing_files(self.selected_model)

    def _task_done_count(self, patient, task: str) -> int:
        """Timepoints already computed for ``task`` against the *current* image set."""
        if task == "ocr":
            if ocr_module.ocr_stale(patient):
                return 0
            return int((patient.load_ocr_times() or {}).get("processed_timepoints", 0) or 0)
        if task == "pred":
            if predmod.predictions_stale(patient):
                return 0
            return int((patient.load_prediction_metadata() or {}).get("computed_timepoints", 0) or 0)
        # bbox: boxes cached at the reference depth
        ref = rcnn_roi.reference_depth(patient)
        if ref is None:
            return 0
        return sum(1 for t in range(patient.num_timepoints()) if patient.has_rcnn_box(ref, t))

    def _task_complete(self, patient, task: str) -> bool:
        if task == "ocr":
            return ocr_module.ocr_complete(patient) and not ocr_module.ocr_stale(patient)
        if task == "pred":
            return predmod.predictions_complete(patient) and not predmod.predictions_stale(patient)
        return not self._patient_needs_bbox(patient)

    def _should_yield_for(self, patient):
        """A ``should_yield`` predicate for ``patient``'s model task: True while a *different*
        embryo is on screen and that on-screen embryo has a live model task (predictions or
        ROI). The on-screen embryo's own task never yields, so it always gets the detector /
        classifier first; priority follows the screen as the user navigates."""
        def should_yield() -> bool:
            cur = self.current_patient_ts
            if cur is None or cur.directory == patient.directory:
                return False
            return (cur.directory, "pred") in self._bg_workers or (cur.directory, "bbox") in self._bg_workers
        return should_yield

    def _bg_job(self, patient, task: str, resume: bool = True):
        """The compute callable for ``(patient, task)``, or None if it can't run yet."""
        if task == "ocr":
            return lambda p, m, c: ocr_module.compute_and_save_ocr_times(patient, progress_cb=p, should_cancel=c)
        if task == "pred":
            if not self._model_ready():
                return None
            model = self.selected_model
            yield_fn = self._should_yield_for(patient)
            return lambda p, m, c: predmod.compute_and_save_predictions(
                patient, model, progress_cb=p, should_cancel=c, should_yield=yield_fn, resume=resume
            )
        if task == "bbox":
            if not setup_models.rcnn_present():
                return None
            yield_fn = self._should_yield_for(patient)
            return lambda p, m, c: rcnn_roi.detect_boxes_for_patient(
                patient, progress_cb=p, should_cancel=c, should_yield=yield_fn
            )
        return None

    def _start_bg_task(self, patient, task: str, resume: bool = True) -> bool:
        """Start a worker for ``(patient, task)`` unless one is already live. Returns True
        if a worker is now running for it. Clears any paused flag (an explicit resume)."""
        key = (patient.directory, task)
        if key in self._bg_workers:
            return True
        job = self._bg_job(patient, task, resume=resume)
        if job is None:
            return False
        self._spawn_bg_worker(patient, task, job)
        return True

    def _spawn_bg_worker(self, patient, task: str, job) -> None:
        """Register, wire, and start a background worker for ``(patient, task)``.

        Any worker already running for the key is cancelled first (a deliberate re-run,
        e.g. choosing a new model). The three bars are updated only when ``patient`` is the
        one on screen; otherwise the work proceeds invisibly and the bar reflects it on
        return."""
        key = (patient.directory, task)
        existing = self._bg_workers.pop(key, None)
        if existing is not None:
            try:
                existing.cancel()
            except RuntimeError:
                pass
            for sig in (existing.progress, existing.message, existing.succeeded, existing.failed):
                try:
                    sig.disconnect()
                except (TypeError, RuntimeError):
                    pass
        self._paused_tasks.discard(key)
        worker = FunctionWorker(job, self)
        self._bg_workers[key] = worker
        self._task_workers.append(worker)  # keep referenced + cancelled on close
        self._bg_progress[key] = (self._task_done_count(patient, task), patient.num_timepoints())
        worker.progress.connect(lambda d, t, k=key: self._on_bg_progress(k, d, t))
        worker.message.connect(lambda msg, k=key: self._on_bg_message(k, msg))
        worker.succeeded.connect(lambda r, k=key, p=patient: self._on_bg_succeeded(k, p))
        worker.failed.connect(lambda e, k=key, p=patient: self._on_bg_failed(k, p, e))
        worker.start()
        if self._is_current_key(key):
            done, total = self._bg_progress[key]
            self._bar_for(task).set_running(done, total, "starting…")
            if task == "pred":
                self._render_roi_bar(patient)

    def _pause_bg_task(self, patient, task: str) -> None:
        """User pause: stop the worker (it flushes partial progress) and mark it sticky.
        Signals are disconnected so the worker's imminent completion can't flip the bar."""
        key = (patient.directory, task)
        worker = self._bg_workers.pop(key, None)
        if worker is not None:
            try:
                worker.cancel()
            except RuntimeError:
                pass
            for sig in (worker.progress, worker.message, worker.succeeded, worker.failed):
                try:
                    sig.disconnect()
                except (TypeError, RuntimeError):
                    pass
        self._paused_tasks.add(key)
        if self._is_current_key(key):
            done = self._task_done_count(patient, task)
            self._bar_for(task).set_idle(f"paused {done}/{patient.num_timepoints()}")
            if task == "pred":
                self._render_roi_bar(patient)

    def autostart_tasks(self, patient) -> None:
        """Auto-start an embryo's not-done / not-paused / not-running tasks (called on
        load/display). Never cancels anything; running tasks for *other* embryos keep going."""
        if patient is None or patient.num_timepoints() == 0:
            return
        for task in ("ocr", "pred"):
            key = (patient.directory, task)
            if key in self._bg_workers or key in self._paused_tasks or self._task_complete(patient, task):
                continue
            self._start_bg_task(patient, task)
        # ROI: predictions produce the boxes when a model is ready; only run the standalone
        # detector when no model is available (so there's never duplicate RCNN work).
        if not self._model_ready():
            key = (patient.directory, "bbox")
            if not (key in self._bg_workers or key in self._paused_tasks or self._task_complete(patient, "bbox")):
                self._start_bg_task(patient, "bbox")

    def refresh_task_bars(self, patient) -> None:
        """Bind the three bars to ``patient``'s current task state (no side effects on
        workers). Auto-start is done separately by :meth:`autostart_tasks`."""
        if patient is None or patient.num_timepoints() == 0:
            for task in self._TASK_BARS:
                self._bar_for(task).set_idle()
            return
        self._render_task_bar(patient, "ocr")
        self._render_task_bar(patient, "pred")
        self._render_roi_bar(patient)

    def _render_task_bar(self, patient, task: str) -> None:
        """Render the OCR or Pred bar from ``patient``'s state."""
        bar = self._bar_for(task)
        key = (patient.directory, task)
        total = patient.num_timepoints()
        if self._task_complete(patient, task):
            tip = None
            if task == "pred":
                tip = f"cached from {patient.load_prediction_metadata().get('model_name', '?')}"
            bar.set_done("cached", tooltip=tip)
            return
        if key in self._bg_workers:
            done, tot = self._bg_progress.get(key, (self._task_done_count(patient, task), total))
            bar.set_running(done, tot, f"{done}/{tot}")
            return
        if key in self._paused_tasks:
            bar.set_idle(f"paused {self._task_done_count(patient, task)}/{total}")
            return
        if task == "pred" and not self.selected_model:
            bar.set_idle("no model")
        elif task == "pred" and setup_models.missing_files(self.selected_model):
            bar.set_error(
                "weights missing",
                tooltip=f"'{self.selected_model}' weights not found — use Tools ▸ Setup Models.",
            )
        else:
            # Runnable but no live worker (e.g. transiently between display and autostart).
            bar.set_idle(f"idle {self._task_done_count(patient, task)}/{total}")

    def _render_roi_bar(self, patient) -> None:
        """Render the ROI bar. When a model is ready, boxes are produced by predictions, so
        the bar *mirrors* the prediction task (advancing per timepoint); otherwise it's a
        standalone detector task with its own worker."""
        total = patient.num_timepoints()
        if self._task_complete(patient, "bbox"):
            self.task_bbox.set_done("cached")
            return
        if self._model_ready():
            # Boxes come from predictions — mirror their state so ROI advances per timepoint.
            pred_key = (patient.directory, "pred")
            done = self._task_done_count(patient, "bbox")
            if pred_key in self._bg_workers:
                self.task_bbox.set_running(done, total, f"{done}/{total} via predictions")
            elif pred_key in self._paused_tasks:
                self.task_bbox.set_idle(f"paused {done}/{total} (via predictions)")
            else:
                self.task_bbox.set_running(done, total, f"{done}/{total} via predictions")
            return
        if not setup_models.rcnn_present():
            self.task_bbox.set_idle("no detector")
            return
        self._render_task_bar_bbox_standalone(patient)

    def _render_task_bar_bbox_standalone(self, patient) -> None:
        key = (patient.directory, "bbox")
        total = patient.num_timepoints()
        done = self._task_done_count(patient, "bbox")
        if key in self._bg_workers:
            d, t = self._bg_progress.get(key, (done, total))
            self.task_bbox.set_running(d, t, f"{d}/{t}")
        elif key in self._paused_tasks:
            self.task_bbox.set_idle(f"paused {done}/{total}")
        else:
            self.task_bbox.set_idle(f"idle {done}/{total}")

    def _refresh_current_assist_views(self) -> None:
        """Re-read the current embryo's OCR/prediction sidecars and repaint the HUD +
        timeline. Used both on task completion and (throttled) as a task fills in."""
        self._invalidate_assist_caches()
        self.update_prediction_label()
        self.update_label_buttons()
        for widget in self._timeline_widgets():
            widget.refresh()

    # --- worker signal handlers (run on the UI thread) -----------------
    def _on_bg_progress(self, key, done: int, total: int) -> None:
        self._bg_progress[key] = (done, total)
        if not self._is_current_key(key):
            return
        self._bar_for(key[1]).set_running(done, total, f"{done}/{total}")
        if key[1] == "pred" and self._model_ready():
            # ROI mirrors predictions: each scored timepoint caches a box. (When the run
            # finishes, _on_bg_succeeded → refresh_task_bars flips the ROI bar to cached.)
            self.task_bbox.set_running(done, total, f"{done}/{total} via predictions")
        # Semi-live: OCR/predictions flush new timepoints to disk every flush interval, so
        # repaint the timeline/HUD at those boundaries (not every tick — matplotlib redraws
        # aren't free) to watch the chart fill in. Completion does a final repaint.
        if key[1] in ("ocr", "pred") and done % self._TIMELINE_REFRESH_EVERY == 0 and done < total:
            self._refresh_current_assist_views()

    def _on_bg_message(self, key, msg: str) -> None:
        if self._is_current_key(key):
            self._bar_for(key[1])._set_state("running", msg, tooltip=msg)

    def _on_bg_succeeded(self, key, patient) -> None:
        self._bg_workers.pop(key, None)
        if self._is_current_key(key):
            self._refresh_current_assist_views()
            if key[1] in ("pred", "bbox"):
                self.refresh_roi_view()
            self.refresh_task_bars(patient)
        # A slot freed (or this embryo finished) → start the next pending background embryo.
        self._schedule_background_work()

    def _on_bg_failed(self, key, patient, error: str) -> None:
        self._bg_workers.pop(key, None)
        if key[1] == "bbox":
            self._roi_detect_failed = True
        if self._is_current_key(key):
            self._bar_for(key[1]).set_error("error", tooltip=error)
        self._schedule_background_work()

    # --- bar click handling --------------------------------------------
    def _on_task_bar_clicked(self, task: str) -> None:
        patient = self.current_patient_ts
        if patient is None or patient.num_timepoints() == 0:
            return
        # ROI in "via predictions" mode isn't an independent task — manage it via Pred.
        if task == "bbox" and self._model_ready() and not self._task_complete(patient, "bbox"):
            return
        key = (patient.directory, task)
        if self._task_complete(patient, task):
            return  # done → no-op
        if key in self._bg_workers:
            self._pause_bg_task(patient, task)
            return
        # paused / idle → resume; surface guidance if it can't run.
        if self._bg_job(patient, task) is None:
            if task == "pred":
                if self._ensure_model_selected() is not None:
                    self._start_bg_task(patient, "pred")
            else:
                QMessageBox.information(
                    self, "No detector",
                    "The RCNN detector (rcnn.pt) isn't installed — use Tools ▸ Setup Models.",
                )
            return
        self._start_bg_task(patient, task)

    def _patient_needs_bbox(self, patient) -> bool:
        depth_name = rcnn_roi.reference_depth(patient)
        if depth_name is None:
            return False
        for timepoint in range(patient.num_timepoints()):
            if not patient.has_rcnn_box(depth_name, timepoint):
                return True
        return False

    # ------------------------------------------------------------------
    # Background sweep over the whole dataset
    #
    # Every loaded embryo's tasks auto-start as their own per-embryo workers (same
    # mechanism as the current embryo), so an embryo doesn't wait until you open it. The
    # current embryo always runs; a bounded number of *other* embryos run alongside it
    # (``_BG_MAX_BACKGROUND``) and the rest start as those finish (``_schedule_background_work``
    # is re-pumped on every task completion). The "All embryos" bar shows how many embryos
    # are fully done and pauses/resumes the whole background sweep.
    # ------------------------------------------------------------------
    def _embryo_needs_work(self, patient) -> bool:
        """True if ``patient`` has a runnable, not-complete, not-user-paused task."""
        if patient.num_timepoints() == 0:
            return False
        for task in self._TASK_BARS:
            if task == "pred" and not self._model_ready():
                continue          # predictions need a model
            if task == "bbox" and self._model_ready():
                continue          # boxes come from predictions when a model is present
            key = (patient.directory, task)
            if key in self._paused_tasks or self._task_complete(patient, task):
                continue
            return True
        return False

    def _schedule_background_work(self) -> None:
        """Keep the current embryo computing and roll a bounded set of other embryos.

        Called on load, navigation, and whenever a task finishes (to fill a freed slot).
        Never cancels anything; honours per-task pauses and the dataset-wide pause."""
        if self.dataset is None:
            return
        if self.current_patient_ts is not None:
            self.autostart_tasks(self.current_patient_ts)  # current: always (priority)
        if not self._bg_paused_all:
            current_dir = self.current_patient_ts.directory if self.current_patient_ts else None
            active_other = {key[0] for key in self._bg_workers if key[0] != current_dir}
            for patient in self.dataset.get_patient_series():
                if len(active_other) >= self._BG_MAX_BACKGROUND:
                    break
                if patient.directory == current_dir or patient.directory in active_other:
                    continue
                if not self._embryo_needs_work(patient):
                    continue
                self.autostart_tasks(patient)
                active_other.add(patient.directory)
        self._update_all_embryos_bar()

    def _update_all_embryos_bar(self) -> None:
        """Render the "All embryos" bar: embryos fully done / total (or paused / disabled)."""
        if self.dataset is None or self.dataset.num_patients() <= 1:
            self.task_prefetch.set_disabled(
                "single embryo",
                tooltip="The dataset-wide sweep runs only when more than one embryo is loaded.",
            )
            return
        patients = [p for p in self.dataset.get_patient_series() if p.num_timepoints() > 0]
        total = len(patients)
        done = sum(1 for p in patients if not self._embryo_needs_work(p))
        if self._bg_paused_all:
            self.task_prefetch.set_idle(f"paused {done}/{total}")
        elif done >= total:
            self.task_prefetch.set_done(f"{total}/{total} embryos")
        else:
            self.task_prefetch.set_running(done, total, f"{done}/{total} embryos")

    def _on_prefetch_bar_clicked(self) -> None:
        """Toggle the dataset-wide background sweep: pause it (stopping non-current
        embryos' workers) or resume it. The current embryo always keeps computing."""
        self._bg_paused_all = not self._bg_paused_all
        if self._bg_paused_all:
            current_dir = self.current_patient_ts.directory if self.current_patient_ts else None
            for key in list(self._bg_workers):
                if key[0] != current_dir:
                    self._cancel_bg_worker(key)  # flushed + resumable; not marked paused
        self._schedule_background_work()

    def _cancel_bg_worker(self, key) -> None:
        """Stop and unwire the worker for ``key`` (no pause flag). It flushes partial
        progress to its sidecar, so a later start resumes from where it stopped."""
        worker = self._bg_workers.pop(key, None)
        if worker is None:
            return
        try:
            worker.cancel()
        except RuntimeError:
            pass
        for sig in (worker.progress, worker.message, worker.succeeded, worker.failed):
            try:
                sig.disconnect()
            except (TypeError, RuntimeError):
                pass

    def _cancel_bg_task(self, patient, task: str) -> None:
        """Stop a worker for ``(patient, task)`` without marking it paused (used when one
        task takes over another's work, e.g. predictions taking over ROI box production)."""
        self._cancel_bg_worker((patient.directory, task))

    def _cancel_all_bg_tasks(self) -> None:
        """Cancel every per-embryo worker and clear task state (used on dataset replace)."""
        for key in list(self._bg_workers):
            self._cancel_bg_worker(key)
        self._bg_progress.clear()
        self._paused_tasks.clear()

    def run_ocr(self) -> None:
        if self.current_patient_ts is None:
            QMessageBox.warning(self, "No patient selected", "Select a patient first.")
            return
        self._start_bg_task(self.current_patient_ts, "ocr")

    def run_predictions(self, force: bool = False) -> None:
        if self.current_patient_ts is None:
            QMessageBox.warning(self, "No patient selected", "Select a patient first.")
            return
        patient = self.current_patient_ts

        if not force and predmod.predictions_complete(patient) and not predmod.predictions_stale(patient):
            self.task_pred.set_done("cached")
            self._invalidate_assist_caches()
            self.update_prediction_label()
            self.update_label_buttons()
            for widget in self._timeline_widgets():
                widget.refresh()
            return

        model_name = self._ensure_model_selected()
        if model_name is None:
            return
        # Predictions detect + cache the ROI box per timepoint, so stand down any standalone
        # ROI detector (the ROI bar then mirrors predictions). ``force`` recomputes from
        # scratch (a fresh worker); otherwise resume/start the sidecar-backed run.
        self._cancel_bg_task(patient, "bbox")
        if force:
            self._spawn_bg_worker(
                patient, "pred",
                lambda p, m, c: predmod.compute_and_save_predictions(
                    patient, model_name, progress_cb=p, should_cancel=c, resume=False
                ),
            )
        else:
            self._start_bg_task(patient, "pred")

    def detect_rois(self) -> None:
        """Compute RCNN ROI boxes for the current embryo (resumable; the one box per
        timepoint is reused across every focal depth). When a model is available the boxes
        come from predictions, so this starts/uses that; otherwise it runs the detector."""
        if self.current_patient_ts is None:
            QMessageBox.warning(self, "No patient selected", "Select a patient first.")
            return
        patient = self.current_patient_ts

        if rcnn_roi.reference_depth(patient) is None:
            QMessageBox.warning(
                self, "No images", "This embryo has no focal-depth images to detect on."
            )
            return
        self._roi_detect_failed = False
        if self._model_ready():
            # Boxes are produced by predictions (no separate detector) — start them.
            self.run_predictions()
            return
        self._start_bg_task(patient, "bbox")

    # ==================================================================
    # Assist data caches + status label
    # ==================================================================
    def _invalidate_assist_caches(self) -> None:
        self._pred_arrays = None
        self._ocr_arrays = None
        self._anomaly_cache = None

    def _get_pred_arrays(self) -> Optional[Dict]:
        if self._pred_arrays is None and self.current_patient_ts is not None:
            self._pred_arrays = predmod.load_prediction_arrays(self.current_patient_ts)
        return self._pred_arrays

    def _get_ocr_arrays(self) -> Optional[Dict]:
        if self._ocr_arrays is None and self.current_patient_ts is not None:
            self._ocr_arrays = ocr_module.load_hours_array(self.current_patient_ts)
        return self._ocr_arrays

    def _get_anomalies(self) -> Optional[Dict]:
        if self._anomaly_cache is None and self.current_patient_ts is not None:
            self._anomaly_cache = anomalies.compute_anomalies(self.current_patient_ts)
        return self._anomaly_cache

    def _timeline_widgets(self) -> List:
        """The embedded strip plus the standalone window when it is open."""
        widgets: List = [self.timeline_strip]
        if self.timeline_window is not None:
            widgets.append(self.timeline_window)
        return widgets

    def _after_label_change(self) -> None:
        self._anomaly_cache = None
        self.update_label_buttons()
        self.update_prediction_label()
        for widget in self._timeline_widgets():
            widget.refresh()

    @staticmethod
    def _chip(text: str, color: str, bold: bool = False) -> str:
        weight = "font-weight:600;" if bold else ""
        return (
            f"<span style='color:{color};{weight}'>{text}</span>"
        )

    def update_prediction_label(self) -> None:
        """Render the HUD: timepoint · OCR hour · prediction · saved label · anomaly."""
        patient = self.current_patient_ts
        if patient is None:
            self.hud_label.setText(
                f"<span style='color:{C_MUTED}'>Load a dataset to begin "
                "(File ▸ Open, or drag a folder in)</span>"
            )
            return

        timepoint = self.current_timepoint
        sep = f" <span style='color:{C_BORDER}'>·</span> "
        parts: List[str] = [self._chip(f"t{timepoint}", C_TEXT, bold=True)]

        ocr_arrays = self._get_ocr_arrays()
        if ocr_arrays and timepoint < len(ocr_arrays["hours"]) and ocr_arrays["hours"][timepoint] is not None:
            suffix = " interp" if timepoint in ocr_arrays["interpolated"] else ""
            parts.append(self._chip(f"{ocr_arrays['hours'][timepoint]:.1f}h{suffix}", C_TEXT))

        pred = self._get_pred_arrays()
        if pred and timepoint < pred["num_timepoints"]:
            shown = pred["post_class"][timepoint] if self.show_postprocessed else pred["raw_class"][timepoint]
            prob = pred["max_prob"][timepoint]
            conf = f" <span style='color:{C_MUTED}'>p={prob:.2f}</span>" if prob is not None else ""
            parts.append(self._chip(f"pred {shown or '—'}", C_PRED, bold=True) + conf)
        else:
            parts.append(self._chip("no prediction", C_MUTED))

        label = patient.get_label(timepoint)
        if label:
            parts.append(self._chip(f"label {label} ✓", C_ACCENT, bold=True))
        else:
            parts.append(self._chip("unlabeled", C_MUTED))

        if self._stage_start is not None or self._stage_end is not None:
            parts.append(self._chip(f"stage [{self._stage_start}, {self._stage_end}]", C_PRED))

        anomaly_data = self._get_anomalies()
        if anomaly_data and timepoint < len(anomaly_data["per_timepoint"]):
            signals = anomaly_data["per_timepoint"][timepoint]["signals"]
            if signals:
                parts.append(self._chip("⚠ " + ", ".join(s["type"] for s in signals), C_ANOM, bold=True))

        self.hud_label.setText(sep.join(parts))

    def toggle_prediction_sequence(self) -> None:
        self.show_postprocessed = not self.show_postprocessed
        self.seq_toggle_btn.blockSignals(True)
        self.seq_toggle_btn.setChecked(self.show_postprocessed)
        self.seq_toggle_btn.setText("postprocessed" if self.show_postprocessed else "raw argmax")
        self.seq_toggle_btn.blockSignals(False)
        self.update_prediction_label()
        self.update_label_buttons()
        for widget in self._timeline_widgets():
            widget.set_show_postprocessed(self.show_postprocessed)

    # ==================================================================
    # Navigation + stage auto-fill (Phase 6)
    # ==================================================================
    def jump_to_timepoint(self, timepoint: int) -> None:
        if self.current_patient_ts is None:
            return
        timepoint = max(0, min(self.current_patient_ts.num_timepoints() - 1, int(timepoint)))
        self.current_timepoint = timepoint
        self.display_image()

    def mark_stage_start(self) -> None:
        if self.current_patient_ts is None:
            return
        self._stage_start = self.current_timepoint
        self.statusBar().showMessage(f"Stage start marked at t={self._stage_start}", 3000)
        self.update_prediction_label()
        for widget in self._timeline_widgets():
            widget.set_stage_markers()

    def mark_stage_end(self) -> None:
        if self.current_patient_ts is None:
            return
        self._stage_end = self.current_timepoint
        self.statusBar().showMessage(f"Stage end marked at t={self._stage_end}", 3000)
        self.update_prediction_label()
        for widget in self._timeline_widgets():
            widget.set_stage_markers()

    def _stage_range(self) -> Optional[tuple]:
        if self._stage_start is None or self._stage_end is None:
            QMessageBox.information(
                self,
                "Mark the stage first",
                "Mark a stage start with '[' and end with ']' (on the timepoint you want), "
                "then fill.",
            )
            return None
        lo, hi = sorted((self._stage_start, self._stage_end))
        return lo, hi

    def fill_stage_with_selected(self) -> None:
        patient = self.current_patient_ts
        if patient is None:
            return
        stage_range = self._stage_range()
        if stage_range is None:
            return
        label = self._selected_label()
        if label is None:
            QMessageBox.information(self, "Select a class", "Select a timepoint class first.")
            return
        lo, hi = stage_range
        for timepoint in range(lo, hi + 1):
            patient.set_label(timepoint, label)
        self.statusBar().showMessage(f"Filled t={lo}..{hi} with {label}", 4000)
        self._after_label_change()

    def accept_predictions_for_stage(self) -> None:
        patient = self.current_patient_ts
        if patient is None:
            return
        stage_range = self._stage_range()
        if stage_range is None:
            return
        pred = self._get_pred_arrays()
        if not pred:
            QMessageBox.information(self, "No predictions", "Run predictions first.")
            return
        lo, hi = stage_range
        applied = 0
        for timepoint in range(lo, hi + 1):
            if timepoint < pred["num_timepoints"]:
                predicted = pred["post_class"][timepoint]
                if predicted:
                    patient.set_label(timepoint, predicted)
                    applied += 1
        self.statusBar().showMessage(f"Accepted predictions for {applied} timepoint(s)", 4000)
        self._after_label_change()

    def accept_prediction_current(self) -> None:
        patient = self.current_patient_ts
        if patient is None:
            return
        pred = self._get_pred_arrays()
        if not pred or self.current_timepoint >= pred["num_timepoints"]:
            QMessageBox.information(self, "No prediction", "No prediction for this timepoint.")
            return
        timepoint = self.current_timepoint
        predicted = pred["post_class"][timepoint] if self.show_postprocessed else pred["raw_class"][timepoint]
        if not predicted:
            return
        patient.set_label(timepoint, predicted)
        self.statusBar().showMessage(f"Accepted '{predicted}' at t={timepoint}", 3000)
        self._after_label_change()

    def goto_next_anomaly(self) -> None:
        self._goto_anomaly(forward=True)

    def goto_prev_anomaly(self) -> None:
        self._goto_anomaly(forward=False)

    def _goto_anomaly(self, forward: bool) -> None:
        if self.current_patient_ts is None:
            return
        anomaly_data = self._get_anomalies()
        flagged = anomaly_data["flagged"] if anomaly_data else []
        if not flagged:
            self.statusBar().showMessage("No anomalies flagged (run predictions/OCR)", 3000)
            return
        current = self.current_timepoint
        if forward:
            candidates = [t for t in flagged if t > current]
            target = candidates[0] if candidates else flagged[0]
        else:
            candidates = [t for t in flagged if t < current]
            target = candidates[-1] if candidates else flagged[-1]
        self.jump_to_timepoint(target)
        self.statusBar().showMessage(
            f"Anomaly at t={target} ({len(flagged)} flagged)", 3000
        )

    def closeEvent(self, event) -> None:
        # Persist any unsaved segmentation strokes before teardown.
        if self.segmentation_window is not None:
            self.segmentation_window._flush()
        # Ask background threads to stop and give them a moment to unwind so Qt doesn't
        # warn about a QThread being destroyed while still running.
        for worker in self._task_workers + ([self._roi_worker] if self._roi_worker else []):
            try:
                worker.cancel()
            except RuntimeError:
                pass
        for worker in self._task_workers + ([self._roi_worker] if self._roi_worker else []):
            try:
                worker.wait(3000)
            except RuntimeError:
                pass
        event.accept()


class AllDepthsWindow(QMainWindow):
    """Dockable window showing all focal depths for the current timepoint."""

    def __init__(self, parent, depths: List[str], patient_ts, timepoint: int):
        super().__init__(parent)
        self.depths = depths
        self.patient_ts = patient_ts
        self.timepoint = timepoint
        self.depth_items: Dict[int, pg.ImageItem] = {}

        self.setWindowTitle("All Focal Depths")
        self.setGeometry(100, 100, 1400, 900)
        self.setStyleSheet(STYLESHEET)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_docks)
        self.layout.addWidget(self.reset_button)

        self.dock_area: Optional[DockArea] = None
        self._build_dock_area()
        self.update_images(patient_ts, timepoint)

    def _build_dock_area(self) -> None:
        if self.dock_area is not None:
            self.layout.removeWidget(self.dock_area)
            self.dock_area.setParent(None)

        self.dock_area = DockArea()
        self.layout.addWidget(self.dock_area)
        self.depth_items.clear()

        docks: List[Dock] = []
        for i, depth in enumerate(self.depths):
            dock = Dock(depth, size=(450, 320), closable=True)
            glw = pg.GraphicsLayoutWidget()
            plot = glw.addPlot(title=depth)
            plot.setAspectLocked(True)
            plot.hideAxis("left")
            plot.hideAxis("bottom")
            plot.getViewBox().invertY(True)

            image_item = pg.ImageItem()
            plot.addItem(image_item)
            dock.addWidget(glw)

            docks.append(dock)
            self.depth_items[i] = image_item

        # Deterministic default: row 1 has 0,1,2; row 2 has 3,4,5; row 3 has 6.
        self.dock_area.addDock(docks[0], "left")
        self.dock_area.addDock(docks[1], "right", docks[0])
        self.dock_area.addDock(docks[2], "right", docks[1])
        self.dock_area.addDock(docks[3], "bottom", docks[0])
        self.dock_area.addDock(docks[4], "right", docks[3])
        self.dock_area.addDock(docks[5], "right", docks[4])
        self.dock_area.addDock(docks[6], "bottom", docks[3])

    def update_images(self, patient_ts, timepoint: int) -> None:
        self.patient_ts = patient_ts
        self.timepoint = timepoint
        for depth_index, image_item in list(self.depth_items.items()):
            if not patient_ts.depth_has_images(depth_index):
                continue
            try:
                image = patient_ts.get_image(timepoint, depth_index)
                image_item.setImage(image.T)
            except RuntimeError:
                # The user may have closed this dock; reset restores it.
                continue

    def reset_docks(self) -> None:
        self._build_dock_area()
        self.update_images(self.patient_ts, self.timepoint)

    def closeEvent(self, event) -> None:
        parent = self.parent()
        if parent is not None:
            parent.all_depths_window = None
            parent.show_all_depths_checkbox.blockSignals(True)
            parent.show_all_depths_checkbox.setChecked(False)
            parent.show_all_depths_checkbox.blockSignals(False)
        event.accept()


class DashboardWindow(QMainWindow):
    """Dataset-level labeling progress dashboard."""

    def __init__(self, parent, dataset: DataSet, classes: List[str]):
        super().__init__(parent)
        self.dataset = dataset
        self.classes = classes
        self.setWindowTitle("Labeling Dashboard")
        self.setGeometry(150, 150, 1300, 1000)
        self.setStyleSheet(STYLESHEET)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.setCentralWidget(self.scroll)

        self.content = QWidget()
        self.layout = QVBoxLayout(self.content)
        self.scroll.setWidget(self.content)

        self._build_dashboard()

    def _build_dashboard(self) -> None:
        summary = self._compute_summary()

        self.layout.addWidget(QLabel(summary["summary_text"]))
        self._add_canvas(self._make_progress_bar_chart(summary))
        self._add_canvas(self._make_class_distribution_chart(summary))
        self._add_canvas(self._make_progress_over_time_chart(summary))

    def _compute_summary(self) -> Dict:
        patient_rows = []
        class_counts = Counter()
        events = []

        total_timepoints = 0
        total_timepoint_labeled = 0

        for patient in self.dataset.get_patient_series():
            n_timepoints = patient.num_timepoints()
            timepoint_labels = patient.labels.get("timepoint_labels", {})

            n_tp = len(timepoint_labels)

            total_timepoints += n_timepoints
            total_timepoint_labeled += n_tp

            for label in timepoint_labels.values():
                class_counts[label] += 1

            for event in patient.label_metadata:
                events.append(event)

            patient_rows.append(
                {
                    "patient_id": patient.patient_id,
                    "num_timepoints": n_timepoints,
                    "timepoint_pct": 100.0 * n_tp / n_timepoints if n_timepoints else 0.0,
                }
            )

        overall_timepoint_pct = 100.0 * total_timepoint_labeled / total_timepoints if total_timepoints else 0.0

        summary_text = (
            f"Dataset: {self.dataset.root_directory}\n"
            f"Patients: {self.dataset.num_patients()} | Timepoints: {total_timepoints}\n"
            f"Timepoint labels complete: {total_timepoint_labeled}/{total_timepoints} "
            f"({overall_timepoint_pct:.1f}%)"
        )

        return {
            "patient_rows": patient_rows,
            "class_counts": class_counts,
            "events": events,
            "summary_text": summary_text,
            "overall_timepoint_pct": overall_timepoint_pct,
        }

    def _make_progress_bar_chart(self, summary: Dict):
        labels = ["OVERALL"] + [row["patient_id"] for row in summary["patient_rows"]]
        timepoint_values = [summary["overall_timepoint_pct"]] + [row["timepoint_pct"] for row in summary["patient_rows"]]

        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x, timepoint_values, 0.6, label="Timepoint label")
        ax.set_title("Labeling Completion by Patient")
        ax.set_ylabel("Complete (%)")
        ax.set_ylim(0, 100)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()
        return fig

    def _make_class_distribution_chart(self, summary: Dict):
        counts = [summary["class_counts"].get(label, 0) for label in self.classes]
        x = np.arange(len(self.classes))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(x, counts)
        ax.set_title("Distribution of Timepoint Labels Across Dataset")
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.set_xticklabels(self.classes, rotation=45, ha="right")
        fig.tight_layout()
        return fig

    def _make_progress_over_time_chart(self, summary: Dict):
        created_events = []
        for event in summary["events"]:
            if event.get("action") != "created":
                continue
            if event.get("event_type") != "timepoint_label":
                continue
            timestamp = event.get("timestamp")
            if not timestamp:
                continue
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                continue
            created_events.append(dt.date())

        fig, ax = plt.subplots(figsize=(12, 4))
        if not created_events:
            ax.set_title("Labeling Progress Over Time")
            ax.text(0.5, 0.5, "No label metadata yet", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            fig.tight_layout()
            return fig

        first_day = min(created_events)
        daily_counts = defaultdict(int)
        for day in created_events:
            daily_counts[(day - first_day).days] += 1

        max_day = max(daily_counts.keys())
        xs = list(range(max_day + 1))
        tp_cumulative = []
        tp_running = 0
        for day_index in xs:
            tp_running += daily_counts[day_index]
            tp_cumulative.append(tp_running)

        ax.plot(xs, tp_cumulative, marker="o", label="Timepoint labels")
        ax.set_title("Cumulative Labeling Progress Over Time")
        ax.set_xlabel("Days since first label was created")
        ax.set_ylabel("Number of labels created")
        ax.legend()
        fig.tight_layout()
        return fig

    def _add_canvas(self, fig) -> None:
        canvas = FigureCanvas(fig)
        self.layout.addWidget(canvas)


def draw_stage_timeline(
    figure,
    app,
    patient,
    *,
    current_timepoint: int,
    show_postprocessed: bool,
    use_ocr_axis: bool,
    compact: bool,
) -> Dict:
    """Draw the stage/OCR timeline onto ``figure`` (dark themed).

    Shared by the embedded strip and the standalone window. x-axis is OCR hours when
    available (else timepoint index); y is the developmental stage. The postprocessed
    sequence is the primary line, with raw argmax shown faintly where it differs; saved
    human labels, anomalies, stage boundaries and the current timepoint are overlaid.
    Returns ``{"x": [...], "current_marker": Line2D | None}`` so the caller can move the
    current-timepoint marker cheaply during navigation.
    """
    figure.clear()
    figure.patch.set_facecolor(C_BG)
    axis = figure.add_subplot(111)
    axis.set_facecolor(C_BG)
    result: Dict = {"x": [], "current_marker": None}

    if patient is None:
        axis.set_axis_off()
        return result

    pred = predmod.load_prediction_arrays(patient)
    ocr_arrays = ocr_module.load_hours_array(patient)

    if pred is None:
        axis.set_axis_off()
        axis.text(
            0.5, 0.5, "Run predictions to populate the timeline",
            ha="center", va="center", transform=axis.transAxes,
            color=C_MUTED, fontsize=9,
        )
        return result

    num_timepoints = pred["num_timepoints"]
    class_names = pred["class_names"]
    name_to_index = {name: i for i, name in enumerate(class_names)}

    using_ocr = bool(ocr_arrays and use_ocr_axis)
    if using_ocr:
        hours = ocr_arrays["hours"]
        x = [
            float(hours[t]) if (t < len(hours) and hours[t] is not None) else float(t)
            for t in range(num_timepoints)
        ]
    else:
        x = [float(t) for t in range(num_timepoints)]
    result["x"] = x

    primary = pred["post_class_index"] if show_postprocessed else pred["raw_class_index"]
    secondary = pred["raw_class_index"] if show_postprocessed else pred["post_class_index"]

    valid = [t for t in range(num_timepoints) if primary[t] is not None]
    if valid:
        axis.step(
            [x[t] for t in valid], [primary[t] for t in valid],
            where="mid", color="#4ea3ff", linewidth=1.6, label="prediction",
        )
    diff = [t for t in range(num_timepoints) if secondary[t] is not None and secondary[t] != primary[t]]
    if diff:
        axis.scatter(
            [x[t] for t in diff], [secondary[t] for t in diff],
            s=16, color=C_MUTED, marker="x", label="alt",
        )

    # Saved human labels (open white circles).
    label_x, label_y = [], []
    for t in range(num_timepoints):
        label = patient.get_label(t)
        if label in name_to_index:
            label_x.append(x[t])
            label_y.append(name_to_index[label])
    if label_x:
        axis.scatter(
            label_x, label_y, s=30, facecolors="none", edgecolors="#ffffff",
            linewidths=1.2, label="saved", zorder=5,
        )

    # Anomalies + stage boundaries.
    anomaly_data = anomalies.compute_anomalies(patient)
    for boundary in anomaly_data["boundaries"]:
        if 0 <= boundary < len(x):
            axis.axvline(x[boundary], color=C_BORDER, linestyle=":", linewidth=0.7, zorder=0)
    flagged = [t for t in anomaly_data["flagged"] if t < len(x)]
    if flagged:
        top = len(class_names) - 0.5
        axis.scatter(
            [x[t] for t in flagged], [top] * len(flagged),
            marker="v", color=C_ANOM, s=26, label="anomaly", zorder=6,
        )

    # Stage auto-fill markers from the main window.
    for marker, color in ((app._stage_start, C_PRED), (app._stage_end, C_ACCENT)):
        if marker is not None and 0 <= marker < len(x):
            axis.axvline(x[marker], color=color, linestyle="--", linewidth=1.2, zorder=1)

    # Current-timepoint marker (moved cheaply on navigation).
    current_x = x[current_timepoint] if 0 <= current_timepoint < len(x) else (x[0] if x else 0)
    result["current_marker"] = axis.axvline(current_x, color=C_ACCENT, linewidth=1.6, alpha=0.85, zorder=4)

    axis.set_yticks(range(len(class_names)))
    axis.set_yticklabels(class_names, fontsize=7 if compact else 9)
    axis.set_ylim(-0.5, len(class_names) - 0.5)
    axis.tick_params(colors=C_TEXT, labelsize=7 if compact else 9)
    for spine in axis.spines.values():
        spine.set_color(C_BORDER)

    if compact:
        axis.set_xlabel("")
        axis.margins(x=0.01)
        figure.subplots_adjust(left=0.075, right=0.995, top=0.96, bottom=0.13)
    else:
        axis.set_xlabel("OCR hours" if using_ocr else "Timepoint index", color=C_TEXT)
        axis.set_ylabel("Stage", color=C_TEXT)
        axis.set_title(f"{patient.patient_id} — click to jump", color=C_TEXT)
        legend = axis.legend(loc="upper left", fontsize=8, facecolor=C_PANEL2, edgecolor=C_BORDER)
        for text in legend.get_texts():
            text.set_color(C_TEXT)
        figure.tight_layout()

    return result


class TimelineView(QWidget):
    """Embeddable stage/OCR timeline. Click to jump the main window to a timepoint."""

    def __init__(self, app: "EmbryoLabelingApp", patient_ts=None, *, compact: bool = False):
        super().__init__()
        self.app = app
        self.patient_ts = patient_ts
        self.compact = compact
        self.show_postprocessed = app.show_postprocessed
        self.use_ocr_axis = True
        self.current_timepoint = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(8, 1.5) if compact else (11, 4.5))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet(f"background: {C_BG};")
        layout.addWidget(self.canvas)
        self.canvas.mpl_connect("button_press_event", self._on_click)

        self._x: List[float] = []
        self._current_marker = None
        self.refresh()

    def set_patient(self, patient_ts) -> None:
        self.patient_ts = patient_ts
        self.current_timepoint = 0
        self.refresh()

    def set_show_postprocessed(self, value: bool) -> None:
        self.show_postprocessed = value
        self.refresh()

    def set_use_ocr_axis(self, value: bool) -> None:
        self.use_ocr_axis = value
        self.refresh()

    def set_stage_markers(self) -> None:
        self.refresh()

    def set_current_timepoint(self, timepoint: int) -> None:
        self.current_timepoint = timepoint
        if self._current_marker is not None and 0 <= timepoint < len(self._x):
            x = self._x[timepoint]
            self._current_marker.set_xdata([x, x])
            self.canvas.draw_idle()

    def refresh(self) -> None:
        result = draw_stage_timeline(
            self.figure,
            self.app,
            self.patient_ts,
            current_timepoint=self.current_timepoint,
            show_postprocessed=self.show_postprocessed,
            use_ocr_axis=self.use_ocr_axis,
            compact=self.compact,
        )
        self._x = result["x"]
        self._current_marker = result["current_marker"]
        self.canvas.draw_idle()

    def _on_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None or not self._x:
            return
        target = min(range(len(self._x)), key=lambda t: abs(self._x[t] - event.xdata))
        self.app.jump_to_timepoint(target)


class TimelineWindow(QMainWindow):
    """Full-size stage/OCR timeline window wrapping a TimelineView with extra controls."""

    def __init__(self, parent: "EmbryoLabelingApp", patient_ts):
        super().__init__(parent)
        self.app = parent
        self.setWindowTitle("Stage / OCR Timeline")
        self.setGeometry(160, 160, 1150, 560)
        self.setStyleSheet(STYLESHEET)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        controls = QHBoxLayout()
        self.postprocess_checkbox = QCheckBox("Postprocessed (else raw argmax)")
        self.postprocess_checkbox.setChecked(parent.show_postprocessed)
        self.postprocess_checkbox.stateChanged.connect(self._on_postprocess_toggled)
        controls.addWidget(self.postprocess_checkbox)

        self.ocr_axis_checkbox = QCheckBox("Use OCR time axis")
        self.ocr_axis_checkbox.setChecked(True)
        self.ocr_axis_checkbox.stateChanged.connect(self._on_ocr_axis_toggled)
        controls.addWidget(self.ocr_axis_checkbox)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.view = TimelineView(parent, patient_ts, compact=False)
        layout.addWidget(self.view)

    # --- API mirrored by the main window's _timeline_widgets() loop ----------
    def set_patient(self, patient_ts) -> None:
        self.view.set_patient(patient_ts)

    def set_current_timepoint(self, timepoint: int) -> None:
        self.view.set_current_timepoint(timepoint)

    def set_show_postprocessed(self, value: bool) -> None:
        self.postprocess_checkbox.blockSignals(True)
        self.postprocess_checkbox.setChecked(value)
        self.postprocess_checkbox.blockSignals(False)
        self.view.set_show_postprocessed(value)

    def set_stage_markers(self) -> None:
        self.view.set_stage_markers()

    def refresh(self) -> None:
        self.view.refresh()

    def _on_postprocess_toggled(self, _state) -> None:
        # Route through the app so the strip, HUD, chips and seq button stay in sync.
        self.app.show_postprocessed = not self.postprocess_checkbox.isChecked()
        self.app.toggle_prediction_sequence()

    def _on_ocr_axis_toggled(self, _state) -> None:
        self.view.set_use_ocr_axis(self.ocr_axis_checkbox.isChecked())

    def closeEvent(self, event) -> None:
        if isinstance(self.app, EmbryoLabelingApp):
            self.app.timeline_window = None
        event.accept()


# ----------------------------------------------------------------------------
# Segmentation pane (pixel-level masks for tPN / tB).
#
# Drawing is adapted from yeastvision's pyqtgraph painting (parts/canvas.py): a brush
# stamps integer class values straight into a numpy mask, which is rendered as a
# translucent color overlay (lut[mask]) on top of the grayscale embryo. The mask is kept
# in row-major (image) orientation so mask[r, c] aligns with get_image(...)[r, c] and the
# saved arrays are directly usable for training. Masks are per-timepoint (the same mask
# covers every focal depth) and per-stage, with exactly one class per pixel.
# ----------------------------------------------------------------------------

# Distinct, legible overlay colors per structure (index 0 = background = transparent).
SEG_STRUCTURE_COLORS: Dict[str, List[str]] = {
    "tPN": ["", "#ff5cc8"],                          # pronucleus — magenta
    "tB":  ["", "#33c4ff", "#ffb000", "#b070ff"],    # TE — cyan, ICM — amber, ZP — violet
}
SEG_OVERLAY_ALPHA = 150          # 0..255 translucency of the painted overlay
SEG_DEFAULT_BRUSH = 8            # brush radius in pixels
SEG_MAX_BRUSH = 120
SEG_BRUSH_STEP = 2
SEG_UNDO_LIMIT = 40              # bounded pre-stroke mask snapshots (memory-safe)
SEG_AUTOSAVE_MS = 600            # debounce window for autosave-on-stroke
# Highlight for a selected connected mass (white frost — distinct from every structure
# color). Ctrl+click (or Select mode) picks the contiguous region under the cursor and
# Backspace/Delete removes it, mirroring yeastvision's click-select-then-delete flow.
SEG_SELECT_RGBA = (255, 255, 255, 120)
SEG_SELECT_CONNECTIVITY = np.ones((3, 3), dtype=int)   # 8-connected "mass"


def _seg_lut(stage: str) -> np.ndarray:
    """RGBA lookup table ``(K, 4)`` mapping each class index to its overlay color.

    Index 0 (background) is fully transparent so the embryo image shows through.
    """
    names = SEGMENTATION_STAGES[stage]
    colors = SEG_STRUCTURE_COLORS[stage]
    lut = np.zeros((len(names), 4), dtype=np.ubyte)
    for index in range(1, len(names)):
        color = QColor(colors[index])
        lut[index] = [color.red(), color.green(), color.blue(), SEG_OVERLAY_ALPHA]
    return lut


class SegViewBox(pg.ViewBox):
    """ViewBox for the segmentation canvas.

    Left-drag is reserved for the mask item (painting), so panning is on right-drag and
    zooming on the wheel — the same button-repurposing idea as yeastvision's
    ``ViewBoxNoRightDrag``, swapped so the primary button paints.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setMenuEnabled(False)
        self.setMouseMode(pg.ViewBox.PanMode)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() in (QtCore.Qt.RightButton, QtCore.Qt.MiddleButton):
            ev.accept()
            # Pan using scene coordinates so the math is correct whether the event was
            # delivered to the ViewBox directly or forwarded from the mask item (whose
            # local frame is in image pixels, not scene pixels).
            current = self.mapSceneToView(ev.scenePos())
            previous = self.mapSceneToView(ev.lastScenePos())
            self.translateBy(x=-(current.x() - previous.x()), y=-(current.y() - previous.y()))
        else:
            # Left/other drags belong to the mask item; don't pan or rubber-band.
            ev.ignore()


class SegImageDraw(pg.ImageItem):
    """Translucent mask overlay that paints class values into the window's numpy mask.

    Adapted from yeastvision's ``ImageDraw``: a precomputed disk brush kernel stamps the
    active class index straight into ``window.mask`` (clipped to image bounds, like
    ``drawAt``), and the overlay re-renders as ``lut[mask]``. Left-drag paints a continuous
    stroke (kernel stamped along the dragged line); right/middle-drag is forwarded to the
    ViewBox for panning. The eraser is just the brush with class index 0.
    """

    def __init__(self, window: "SegmentationWindow"):
        super().__init__()
        self.window = window
        self.setOpts(axisOrder="row-major")
        self.setZValue(10)
        self._last_pos = None

    def render_mask(self) -> None:
        mask = self.window.mask
        lut = self.window.lut
        if mask is None:
            self.clear()
            return
        safe = np.clip(mask, 0, len(lut) - 1)
        self.setImage(lut[safe], autoLevels=False)

    @staticmethod
    def _is_select(ev) -> bool:
        return bool(ev.modifiers() & QtCore.Qt.ControlModifier)

    # --- mouse ------------------------------------------------------------
    def mouseClickEvent(self, ev):
        if self.window.mask is None or ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        ev.accept()
        if self.window.select_mode or self._is_select(ev):
            self.window.select_mass_at(int(ev.pos().y()), int(ev.pos().x()))
            return
        self.window.begin_stroke()
        self._stamp_at(int(ev.pos().y()), int(ev.pos().x()))
        self.render_mask()
        self.window.end_stroke()

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton or self.window.mask is None:
            # Right/middle drag pans; hand the event to the ViewBox.
            self.window.view.mouseDragEvent(ev)
            return
        if self.window.select_mode or self._is_select(ev):
            # Selecting, not painting: pick the mass under the drag start, ignore the rest.
            ev.accept()
            if ev.isStart():
                self.window.select_mass_at(int(ev.pos().y()), int(ev.pos().x()))
            return
        ev.accept()
        if ev.isStart():
            self.window.begin_stroke()
            self._stamp_at(int(ev.pos().y()), int(ev.pos().x()))
            self._last_pos = ev.pos()
        elif ev.isFinish():
            self._last_pos = None
            self.window.end_stroke()
        else:
            if self._last_pos is not None:
                self._stamp_line(self._last_pos, ev.pos())
            self._last_pos = ev.pos()
        self.render_mask()

    # --- brush stamping ---------------------------------------------------
    def _stamp_line(self, p0, p1) -> None:
        from skimage.draw import line

        rr, cc = line(int(p0.y()), int(p0.x()), int(p1.y()), int(p1.x()))
        for y, x in zip(rr, cc):
            self._stamp_at(int(y), int(x))

    def _stamp_at(self, cy: int, cx: int) -> None:
        mask = self.window.mask
        kernel = self.window.brush_kernel
        height, width = mask.shape
        kh, kw = kernel.shape
        r0, c0 = cy - kh // 2, cx - kw // 2
        # Intersect the kernel footprint with the image bounds (mirrors drawAt's clip).
        mr0, mc0 = max(0, r0), max(0, c0)
        mr1, mc1 = min(height, r0 + kh), min(width, c0 + kw)
        if mr0 >= mr1 or mc0 >= mc1:
            return
        sub = kernel[mr0 - r0: mr1 - r0, mc0 - c0: mc1 - c0]
        mask[mr0:mr1, mc0:mc1][sub] = self.window.current_class_index


class SegmentationWindow(QMainWindow):
    """Separate keyboard-first pane for painting pixel-level masks at tPN / tB frames.

    Shares patient/image state with the main window like the other auxiliary windows
    (``AllDepthsWindow`` / ``TimelineWindow``): the app holds a reference and repoints it
    via :meth:`set_patient` on patient change. All filesystem access goes through
    ``data.PatientTimeSeries``; this window only paints into an in-memory numpy mask and
    calls the durable save/load APIs.
    """

    def __init__(self, app: "EmbryoLabelingApp", patient_ts, *, timepoint: int = 0, depth: Optional[int] = None):
        super().__init__(app)
        self.app = app
        self.patient_ts = patient_ts
        self.current_timepoint = timepoint
        self.current_depth = (
            depth if depth is not None
            else (patient_ts.default_depth_index() if patient_ts else 0)
        )
        self.active_stage = next(iter(SEGMENTATION_STAGES))
        self.current_class_index = 1
        self.brush_size = SEG_DEFAULT_BRUSH
        self.brush_kernel = self._make_kernel(self.brush_size)
        self.lut = _seg_lut(self.active_stage)
        self.mask: Optional[np.ndarray] = None

        self._dirty = False
        self._loading = False
        self._undo: List[np.ndarray] = []
        self._redo: List[np.ndarray] = []
        self.structure_buttons = QButtonGroup(self)
        self.class_by_button: Dict[QPushButton, int] = {}

        # Mass-selection state (Ctrl+click / Select mode → Backspace deletes).
        self.select_mode = False
        self._selection: Optional[np.ndarray] = None   # boolean (H, W) union of selected masses

        self.setWindowTitle("Segmentation")
        self.setGeometry(140, 140, 1180, 900)
        self.setStyleSheet(STYLESHEET)

        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(SEG_AUTOSAVE_MS)
        self._autosave_timer.timeout.connect(self._flush)

        self._build_ui()
        self._build_shortcuts()
        self._apply_stage(self.active_stage)   # builds structure chips + lut

        if patient_ts is not None and patient_ts.num_timepoints() > 0:
            self._load_mask_for_current()
            self._render_base()
            self.view.autoRange()
        self._update_indicators()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    @staticmethod
    def _make_kernel(radius: int) -> np.ndarray:
        from skimage.morphology import disk

        return disk(max(0, int(radius))).astype(bool)

    def _vsep(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.VLine)
        line.setStyleSheet(f"color: {C_BORDER};")
        return line

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        root.addWidget(self._build_toolbar())

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(C_IMG_BG)
        self.view = SegViewBox()
        self.view.setAspectLocked(True)
        self.view.invertY(True)
        self.glw.addItem(self.view)
        self.base_item = pg.ImageItem()
        self.base_item.setOpts(axisOrder="row-major")
        self.mask_item = SegImageDraw(self)
        self.selection_item = pg.ImageItem()
        self.selection_item.setOpts(axisOrder="row-major")
        self.selection_item.setZValue(20)   # selection highlight sits above the mask
        self.view.addItem(self.base_item)
        self.view.addItem(self.mask_item)
        self.view.addItem(self.selection_item)
        root.addWidget(self.glw, 1)

        self.hint_label = QLabel(
            "left-drag paint · right-drag pan · wheel zoom    |    "
            "1/2/3 structure · 0/E erase · [ ] brush · Ctrl+Z/Ctrl+Shift+Z undo/redo · "
            "Ctrl+click (or V) select mass · ⌫ delete · Esc clear · "
            "← → frame · ↑ ↓ depth · P/B jump to tPN/tB · S switch stage"
        )
        self.hint_label.setStyleSheet(f"color: {C_MUTED}; font-size: 10px;")
        root.addWidget(self.hint_label)

    def _build_toolbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("Header")
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(8)

        # Active-stage selector.
        self.stage_buttons = QButtonGroup(self)
        for stage in SEGMENTATION_STAGES:
            btn = QToolButton()
            btn.setText(stage)
            btn.setCheckable(True)
            btn.setToolTip(f"Segment the {stage} stage")
            btn.clicked.connect(lambda _c, s=stage: self.set_stage(s))
            self.stage_buttons.addButton(btn)
            row.addWidget(btn)
        row.addWidget(self._vsep())

        # Predictions-driven jumps.
        self.jump_pn_btn = QToolButton()
        self.jump_pn_btn.setText("⤓ tPN")
        self.jump_pn_btn.setToolTip("Jump to the first predicted tPN frame (P)")
        self.jump_pn_btn.clicked.connect(lambda: self.jump_to_stage("tPN"))
        self.jump_b_btn = QToolButton()
        self.jump_b_btn.setText("⤓ tB")
        self.jump_b_btn.setToolTip("Jump to the first predicted tB frame (B)")
        self.jump_b_btn.clicked.connect(lambda: self.jump_to_stage("tB"))
        row.addWidget(self.jump_pn_btn)
        row.addWidget(self.jump_b_btn)
        row.addWidget(self._vsep())

        # Frame + focal-depth navigation.
        prev_tp = QToolButton()
        prev_tp.setText("◂")
        prev_tp.setToolTip("Previous timepoint (←)")
        prev_tp.clicked.connect(self.prev_timepoint)
        self.tp_indicator = QLabel("t —")
        self.tp_indicator.setMinimumWidth(104)
        self.tp_indicator.setAlignment(Qt.AlignCenter)
        self.tp_indicator.setStyleSheet("font-weight: 600;")
        next_tp = QToolButton()
        next_tp.setText("▸")
        next_tp.setToolTip("Next timepoint (→)")
        next_tp.clicked.connect(self.next_timepoint)
        row.addWidget(prev_tp)
        row.addWidget(self.tp_indicator)
        row.addWidget(next_tp)

        prev_d = QToolButton()
        prev_d.setText("▾")
        prev_d.setToolTip("Lower focal depth (↓)")
        prev_d.clicked.connect(self.prev_depth)
        self.depth_indicator = QLabel("—")
        self.depth_indicator.setMinimumWidth(48)
        self.depth_indicator.setAlignment(Qt.AlignCenter)
        next_d = QToolButton()
        next_d.setText("▴")
        next_d.setToolTip("Higher focal depth (↑)")
        next_d.clicked.connect(self.next_depth)
        row.addWidget(prev_d)
        row.addWidget(self.depth_indicator)
        row.addWidget(next_d)

        row.addStretch(1)

        # Structure chips (rebuilt per active stage).
        self.structure_bar = QHBoxLayout()
        self.structure_bar.setSpacing(5)
        row.addLayout(self.structure_bar)
        row.addWidget(self._vsep())

        # Brush size.
        row.addWidget(QLabel("brush"))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, SEG_MAX_BRUSH)
        self.brush_spin.setValue(self.brush_size)
        self.brush_spin.setToolTip("Brush radius in pixels ([ / ])")
        self.brush_spin.valueChanged.connect(self.set_brush_size)
        row.addWidget(self.brush_spin)

        # Select mass / undo / redo / clear.
        self.select_btn = QToolButton()
        self.select_btn.setText("⊙ select")
        self.select_btn.setCheckable(True)
        self.select_btn.setToolTip(
            "Select connected masses to delete (or Ctrl+click); ⌫ deletes, Esc clears (V)"
        )
        self.select_btn.clicked.connect(lambda checked: self.toggle_select_mode(checked))
        undo_btn = QToolButton()
        undo_btn.setText("↶")
        undo_btn.setToolTip("Undo (Ctrl+Z)")
        undo_btn.clicked.connect(self.undo)
        redo_btn = QToolButton()
        redo_btn.setText("↷")
        redo_btn.setToolTip("Redo (Ctrl+Shift+Z)")
        redo_btn.clicked.connect(self.redo)
        clear_btn = QToolButton()
        clear_btn.setText("clear")
        clear_btn.setToolTip("Clear this frame's mask")
        clear_btn.clicked.connect(self.clear_mask)
        row.addWidget(self.select_btn)
        row.addWidget(undo_btn)
        row.addWidget(redo_btn)
        row.addWidget(clear_btn)

        row.addWidget(self._vsep())
        self.save_indicator = QLabel("")
        self.save_indicator.setMinimumWidth(78)
        self.save_indicator.setAlignment(Qt.AlignCenter)
        row.addWidget(self.save_indicator)
        return bar

    def _build_shortcuts(self) -> None:
        def add(key, fn):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(fn)
            return shortcut

        self._shortcuts = [
            add(Qt.Key_Left, self.prev_timepoint),
            add(Qt.Key_Right, self.next_timepoint),
            add(Qt.Key_Up, self.next_depth),
            add(Qt.Key_Down, self.prev_depth),
            add("[", lambda: self._bump_brush(-SEG_BRUSH_STEP)),
            add("]", lambda: self._bump_brush(SEG_BRUSH_STEP)),
            add("Ctrl+Z", self.undo),
            add("Ctrl+Shift+Z", self.redo),
            add("Ctrl+Y", self.redo),
            add("Ctrl+S", self._flush),
            add("S", self.toggle_stage),
            add("V", self.toggle_select_mode),
            add("P", lambda: self.jump_to_stage("tPN")),
            add("B", lambda: self.jump_to_stage("tB")),
            add("E", lambda: self.select_class(0)),
        ]
        for digit in range(4):
            self._shortcuts.append(add(str(digit), lambda d=digit: self.select_class(d)))

    # ------------------------------------------------------------------
    # Stage / structure / brush selection
    # ------------------------------------------------------------------
    def _make_structure_chip(self, text: str, color: Optional[str], index: int) -> QPushButton:
        chip = QPushButton(text)
        chip.setCheckable(True)
        chip.setCursor(Qt.PointingHandCursor)
        base = (
            f"QPushButton {{ text-align:left; padding:4px 10px; border-radius:6px;"
            f" border:1px solid {C_BORDER}; background:{C_PANEL2};"
        )
        if color:
            chip.setStyleSheet(
                base + f" color:{C_TEXT}; }}"
                f"QPushButton:checked {{ background:{color}; border-color:{color};"
                " color:#10131a; font-weight:700; }"
            )
        else:
            chip.setStyleSheet(
                base + f" color:{C_MUTED}; }}"
                f"QPushButton:checked {{ background:{C_HOVER}; color:{C_TEXT}; font-weight:600; }}"
            )
        chip.clicked.connect(lambda _c, i=index: self.select_class(i))
        self.structure_buttons.addButton(chip)
        self.class_by_button[chip] = index
        return chip

    def _rebuild_structure_chips(self) -> None:
        for chip in list(self.class_by_button):
            self.structure_buttons.removeButton(chip)
            chip.setParent(None)
            chip.deleteLater()
        self.class_by_button.clear()
        while self.structure_bar.count():
            item = self.structure_bar.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        names = SEGMENTATION_STAGES[self.active_stage]
        colors = SEG_STRUCTURE_COLORS[self.active_stage]
        for index in range(1, len(names)):
            self.structure_bar.addWidget(
                self._make_structure_chip(f"{index}  {names[index]}", colors[index], index)
            )
        self.structure_bar.addWidget(self._make_structure_chip("0  erase", None, 0))
        self.select_class(1 if len(names) > 1 else 0)

    def select_class(self, index: int) -> None:
        names = SEGMENTATION_STAGES[self.active_stage]
        if not (0 <= index < len(names)):
            return
        self.current_class_index = index
        for chip, idx in self.class_by_button.items():
            chip.setChecked(idx == index)

    def set_brush_size(self, value: int) -> None:
        self.brush_size = max(1, min(SEG_MAX_BRUSH, int(value)))
        self.brush_kernel = self._make_kernel(self.brush_size)
        if self.brush_spin.value() != self.brush_size:
            self.brush_spin.blockSignals(True)
            self.brush_spin.setValue(self.brush_size)
            self.brush_spin.blockSignals(False)

    def _bump_brush(self, delta: int) -> None:
        self.set_brush_size(self.brush_size + delta)

    def _apply_stage(self, stage: str) -> None:
        self.active_stage = stage
        self.lut = _seg_lut(stage)
        for btn in self.stage_buttons.buttons():
            btn.setChecked(btn.text() == stage)
        self._rebuild_structure_chips()

    def set_stage(self, stage: str) -> None:
        if stage not in SEGMENTATION_STAGES:
            return
        if stage == self.active_stage and self.mask is not None:
            for btn in self.stage_buttons.buttons():
                btn.setChecked(btn.text() == stage)
            return
        self._flush()
        self._apply_stage(stage)
        self._load_mask_for_current()   # same frame, different stage → different mask file
        self._update_indicators()

    def toggle_stage(self) -> None:
        stages = list(SEGMENTATION_STAGES)
        nxt = stages[(stages.index(self.active_stage) + 1) % len(stages)]
        self.set_stage(nxt)

    # ------------------------------------------------------------------
    # Image + mask display
    # ------------------------------------------------------------------
    def _render_base(self) -> None:
        patient = self.patient_ts
        if patient is None or not patient.depth_has_images(self.current_depth):
            self.base_item.clear()
            return
        image = patient.get_image(self.current_timepoint, self.current_depth)
        self.base_item.setImage(image, autoLevels=True)

    def _load_mask_for_current(self) -> None:
        self._loading = True
        try:
            patient = self.patient_ts
            if (
                patient is None
                or patient.num_timepoints() == 0
                or not patient.depth_has_images(self.current_depth)
            ):
                self.mask = None
                self._selection = None
                self.mask_item.clear()
                self.selection_item.clear()
                return
            image = patient.get_image(self.current_timepoint, self.current_depth)
            height, width = image.shape[:2]
            mask, _meta = patient.load_segmentation_mask(self.active_stage, self.current_timepoint)
            if mask is None:
                mask = np.zeros((height, width), dtype=np.uint8)
            elif mask.shape != (height, width):
                QMessageBox.warning(
                    self,
                    "Mask size mismatch",
                    f"The saved {self.active_stage} mask for t={self.current_timepoint} is "
                    f"{mask.shape[1]}×{mask.shape[0]} but this image is {width}×{height}. "
                    "Starting from a blank mask; the saved file is kept until you draw.",
                )
                mask = np.zeros((height, width), dtype=np.uint8)
            self.mask = mask
            self._undo.clear()
            self._redo.clear()
            self._selection = None
            self._dirty = False
            self.mask_item.render_mask()
            self.selection_item.clear()
        finally:
            self._loading = False
        self._update_save_indicator()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def prev_timepoint(self) -> None:
        self._goto_timepoint(self.current_timepoint - 1)

    def next_timepoint(self) -> None:
        self._goto_timepoint(self.current_timepoint + 1)

    def _goto_timepoint(self, timepoint: int) -> None:
        patient = self.patient_ts
        if patient is None or patient.num_timepoints() == 0:
            return
        timepoint = max(0, min(patient.num_timepoints() - 1, int(timepoint)))
        if timepoint == self.current_timepoint:
            return
        self._flush()
        self.current_timepoint = timepoint
        self._load_mask_for_current()
        self._render_base()
        self._update_indicators()

    def prev_depth(self) -> None:
        self._goto_depth(-1)

    def next_depth(self) -> None:
        self._goto_depth(1)

    def _goto_depth(self, step: int) -> None:
        patient = self.patient_ts
        if patient is None:
            return
        index = self.current_depth + step
        while 0 <= index < patient.num_depths():
            if patient.depth_has_images(index):
                self.current_depth = index
                # Mask is per-timepoint (depth-agnostic): only the background changes.
                self._render_base()
                self._update_indicators()
                return
            index += step

    def jump_to_stage(self, stage: str) -> None:
        patient = self.patient_ts
        if patient is None:
            return
        pred = predmod.load_prediction_arrays(patient)
        if not pred:
            QMessageBox.information(
                self,
                "No predictions",
                "No stage predictions for this patient yet. Run them from the main window "
                "(Tools ▸ Run Predictions), or use the arrow keys to scroll to the frame.",
            )
            return
        post = pred["post_class"]
        target = next(
            (t for t in range(pred["num_timepoints"]) if post[t] == stage), None
        )
        if target is None:
            QMessageBox.information(
                self,
                f"No predicted {stage}",
                f"No timepoint was predicted as {stage} for this patient. "
                "Use the arrow keys to scroll to the frame manually if needed.",
            )
            return
        self._flush()
        self._apply_stage(stage)
        self.current_timepoint = target
        self._load_mask_for_current()
        self._render_base()
        self._update_indicators()
        self.statusBar().showMessage(f"Jumped to predicted {stage} at t={target}", 4000)

    # ------------------------------------------------------------------
    # Stroke / undo / redo
    # ------------------------------------------------------------------
    def _push_undo(self) -> None:
        """Snapshot the current mask onto the undo stack (drops the redo stack)."""
        self._redo.clear()
        if self.mask is not None:
            self._undo.append(self.mask.copy())
            if len(self._undo) > SEG_UNDO_LIMIT:
                self._undo.pop(0)

    def begin_stroke(self) -> None:
        """Start a paint stroke: snapshot for undo and drop any mass selection."""
        self._push_undo()
        self.clear_selection()

    def end_stroke(self) -> None:
        self._mark_dirty()

    def undo(self) -> None:
        if not self._undo or self.mask is None:
            return
        self._redo.append(self.mask.copy())
        self.mask = self._undo.pop()
        self.clear_selection()
        self.mask_item.render_mask()
        self._mark_dirty()

    def redo(self) -> None:
        if not self._redo or self.mask is None:
            return
        self._undo.append(self.mask.copy())
        self.mask = self._redo.pop()
        self.clear_selection()
        self.mask_item.render_mask()
        self._mark_dirty()

    def clear_mask(self) -> None:
        if self.mask is None or not self.mask.any():
            return
        self.begin_stroke()
        self.mask[:] = 0
        self.mask_item.render_mask()
        self._mark_dirty()

    # ------------------------------------------------------------------
    # Mass selection + delete (Ctrl+click / Select mode → Backspace)
    # ------------------------------------------------------------------
    def toggle_select_mode(self, on: Optional[bool] = None) -> None:
        self.select_mode = (not self.select_mode) if on is None else bool(on)
        if self.select_btn.isChecked() != self.select_mode:
            self.select_btn.blockSignals(True)
            self.select_btn.setChecked(self.select_mode)
            self.select_btn.blockSignals(False)
        if not self.select_mode:
            self.clear_selection()
        self.statusBar().showMessage(
            "Select mode ON — click masses, ⌫ to delete, Esc to clear"
            if self.select_mode else "Select mode off",
            3000,
        )

    @staticmethod
    def _connected_component(mask: np.ndarray, y: int, x: int) -> np.ndarray:
        """Boolean (H, W) of the 8-connected region of same-class pixels containing (y, x)."""
        from scipy import ndimage

        labeled, _ = ndimage.label(mask == mask[y, x], structure=SEG_SELECT_CONNECTIVITY)
        return labeled == labeled[y, x]

    def select_mass_at(self, y: int, x: int) -> None:
        mask = self.mask
        if mask is None:
            return
        height, width = mask.shape
        if not (0 <= y < height and 0 <= x < width):
            return
        if int(mask[y, x]) == 0:
            self.clear_selection()   # clicking background clears the selection
            return
        component = self._connected_component(mask, y, x)
        if self._selection is None:
            self._selection = np.zeros((height, width), dtype=bool)
        if self._selection[y, x]:        # toggle this mass off if already selected
            self._selection &= ~component
        else:
            self._selection |= component
        if not self._selection.any():
            self._selection = None
        self._render_selection()
        self._update_indicators()
        count = self._selection_count()
        self.statusBar().showMessage(
            f"{count} mass(es) selected — ⌫ to delete" if count else "Selection cleared", 3000
        )

    def _selection_count(self) -> int:
        if self._selection is None or not self._selection.any():
            return 0
        from scipy import ndimage

        _labeled, n = ndimage.label(self._selection, structure=SEG_SELECT_CONNECTIVITY)
        return int(n)

    def _render_selection(self) -> None:
        if self._selection is None or not self._selection.any():
            self.selection_item.clear()
            return
        highlight = np.zeros((*self._selection.shape, 4), dtype=np.uint8)
        highlight[self._selection] = SEG_SELECT_RGBA
        self.selection_item.setImage(highlight, autoLevels=False)

    def clear_selection(self) -> None:
        if self._selection is not None:
            self._selection = None
            self._render_selection()
            self._update_indicators()

    def delete_selection(self) -> None:
        if self.mask is None or self._selection is None or not self._selection.any():
            return
        count = self._selection_count()
        self._push_undo()                # undoable; keeps the selection until after delete
        self.mask[self._selection] = 0
        self._selection = None
        self.mask_item.render_mask()
        self._render_selection()
        self._mark_dirty()
        self._update_indicators()
        self.statusBar().showMessage(f"Deleted {count} mass(es)", 3000)

    def keyPressEvent(self, event) -> None:
        # Backspace/Delete remove the selected masses; Esc clears the selection. Handled
        # here (not as QShortcuts) so Backspace still edits the brush spin box when focused.
        if event.key() in (Qt.Key_Backspace, Qt.Key_Delete):
            if self._selection is not None and self._selection.any():
                self.delete_selection()
                event.accept()
                return
        elif event.key() == Qt.Key_Escape and self._selection is not None:
            self.clear_selection()
            event.accept()
            return
        super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Persistence (debounced autosave + synchronous flush)
    # ------------------------------------------------------------------
    def _mark_dirty(self) -> None:
        if self._loading:
            return
        self._dirty = True
        self._update_save_indicator()
        self._autosave_timer.start()   # restart the debounce window

    def _flush(self) -> None:
        """Write the current frame's mask now (also called before any navigation)."""
        if not self._dirty or self.patient_ts is None or self.mask is None:
            return
        self._autosave_timer.stop()
        try:
            self.patient_ts.save_segmentation_mask(
                self.active_stage, self.current_timepoint, self.mask
            )
            self._dirty = False
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", f"Could not save mask:\n{exc}")
            return
        self._update_save_indicator()
        self._update_indicators()

    # ------------------------------------------------------------------
    # Indicators + patient swap
    # ------------------------------------------------------------------
    def _update_indicators(self) -> None:
        patient = self.patient_ts
        if patient is None or patient.num_timepoints() == 0:
            self.tp_indicator.setText("t —")
            self.depth_indicator.setText("—")
            self._update_save_indicator()
            return
        drawn = self.mask is not None and bool(self.mask.any())
        has_saved = patient.has_segmentation_mask(self.active_stage, self.current_timepoint)
        flag = " ●" if (drawn or has_saved) else ""
        self.tp_indicator.setText(
            f"{self.active_stage} · t {self.current_timepoint}/{patient.num_timepoints() - 1}{flag}"
        )
        self.depth_indicator.setText(patient.get_depth_name(self.current_depth))
        self._update_save_indicator()

    def _update_save_indicator(self) -> None:
        if self.mask is None:
            self.save_indicator.setText("")
            return
        if self._dirty:
            self.save_indicator.setText("● unsaved")
            self.save_indicator.setStyleSheet(f"color: {C_PRED};")
        elif self.patient_ts is not None and self.patient_ts.has_segmentation_mask(
            self.active_stage, self.current_timepoint
        ):
            self.save_indicator.setText("✓ saved")
            self.save_indicator.setStyleSheet(f"color: {C_ACCENT};")
        else:
            self.save_indicator.setText("empty")
            self.save_indicator.setStyleSheet(f"color: {C_MUTED};")

    def set_patient(self, patient_ts) -> None:
        self._flush()
        self.patient_ts = patient_ts
        self.current_timepoint = 0
        self.current_depth = patient_ts.default_depth_index() if patient_ts else 0
        self._undo.clear()
        self._redo.clear()
        self._selection = None
        self._dirty = False
        if patient_ts is not None and patient_ts.num_timepoints() > 0:
            self._load_mask_for_current()
            self._render_base()
            self.view.autoRange()
        else:
            self.mask = None
            self.base_item.clear()
            self.mask_item.clear()
            self.selection_item.clear()
        self._update_indicators()

    def closeEvent(self, event) -> None:
        self._flush()
        if isinstance(self.app, EmbryoLabelingApp):
            self.app.segmentation_window = None
        event.accept()


def configure_image_view(image_view: pg.ImageView) -> None:
    """Display-only image view: no axes, no histogram, no menu, fit-to-content.

    Killing the pixel-coordinate axes and the LUT histogram is what reclaims the wasted
    space around the embryo; aspect is locked so the image is never stretched.
    """
    image_view.ui.roiBtn.hide()
    image_view.ui.menuBtn.hide()
    image_view.ui.histogram.hide()

    view = image_view.getView()
    view.setMenuEnabled(False)
    view.setAspectLocked(True)
    view.hideAxis("left")
    view.hideAxis("bottom")
    try:
        image_view.ui.graphicsView.setBackground(C_IMG_BG)
    except Exception:
        pass


if __name__ == "__main__":
    pg.setConfigOption("background", C_IMG_BG)
    pg.setConfigOption("foreground", C_TEXT)
    app = QApplication([])
    apply_dark_palette(app)
    window = EmbryoLabelingApp()
    window.show()
    app.exec_()
