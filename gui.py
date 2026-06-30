"""PyQt5 + pyqtgraph desktop GUI for embryo timepoint labeling."""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pyqtgraph.Qt import QtCore
from pyqtgraph.dockarea.Dock import Dock
from pyqtgraph.dockarea.DockArea import DockArea
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QKeySequence, QPalette
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

import anomalies
import ocr as ocr_module
import predictions as predmod
import rcnn_roi
import setup_models
from data import DataSet, ORDERED_FOCAL_DEPTH

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
/* Depth chips */
QPushButton[chip="depth"] {{
    padding: 4px 0; border-radius: 6px; border: 1px solid {C_BORDER};
    background: {C_PANEL2}; color: {C_MUTED}; font-size: 11px;
}}
QPushButton[chip="depth"]:hover {{ background: {C_HOVER}; }}
QPushButton[chip="depth"]:checked {{
    background: {C_ACCENT}; border-color: {C_ACCENT}; color: #0c1f0c; font-weight: 700;
}}
QPushButton[chip="depth"][current="true"] {{ border: 2px solid {C_TEXT}; color: {C_TEXT}; }}
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


class TaskDot(QWidget):
    """Compact background-task indicator for the header: a colored dot + label.

    Exposes the same surface the app uses to drive a task row — ``start`` wires a
    :class:`FunctionWorker`'s signals, and ``worker`` reflects the live worker so the
    app can tell whether a task is running. Clicking an active dot cancels it.
    """

    _ACTIVE = C_ACCENT
    _IDLE = C_MUTED
    _ERROR = C_ANOM

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self._name = name
        self.dot = QLabel("●")
        self.dot.setStyleSheet(f"color: {self._IDLE}; font-size: 13px;")
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        layout.addWidget(self.dot)
        layout.addWidget(self.name_label)
        self.worker: Optional[FunctionWorker] = None
        self.setCursor(Qt.PointingHandCursor)
        self._set("idle", self._IDLE)

    def _set(self, status: str, color: str, tooltip: Optional[str] = None) -> None:
        self.dot.setStyleSheet(f"color: {color}; font-size: 13px;")
        self.setToolTip(f"{self._name}: {tooltip or status}")

    def mousePressEvent(self, event) -> None:
        if self.worker is not None:
            self._cancel_clicked()
        super().mousePressEvent(event)

    # --- worker wiring (mirrors the old TaskProgressWidget contract) ---------
    def start(self, worker: FunctionWorker, on_success=None, on_failed=None) -> None:
        self.cancel()
        self.worker = worker
        self._set("starting…", self._ACTIVE)

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
            self._set(f"{done}/{total}", self._ACTIVE)

    def _on_message(self, text: str) -> None:
        self._set(text, self._ACTIVE, tooltip=text)

    def _finish_done(self) -> None:
        self._set("done", self._ACTIVE)
        self.worker = None

    def _finish_error(self, error: str) -> None:
        self._set("error", self._ERROR, tooltip=error)
        self.worker = None

    def _cancel_clicked(self) -> None:
        self.cancel()
        self._set("cancelled", self._IDLE)

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
        self._set(text, self._IDLE)

    def set_done(self, text: str = "done", tooltip: Optional[str] = None) -> None:
        self._set(text, self._ACTIVE, tooltip=tooltip)

    def set_error(self, text: str, tooltip: Optional[str] = None) -> None:
        self._set(text, self._ERROR, tooltip=tooltip)


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


class EmbryoLabelingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.dataset: Optional[DataSet] = None
        self.current_patient_ts = None
        self.current_timepoint = 0
        self.current_depth = 0
        self.selected_focal_depths: List[str] = []
        self.all_depths_window: Optional[AllDepthsWindow] = None
        self.dashboard_window: Optional[DashboardWindow] = None
        self.timeline_window: Optional["TimelineWindow"] = None
        self._updating_best_depth_checks = False

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

        # Background-work bookkeeping (workers kept alive until they finish).
        self._workers: List[FunctionWorker] = []
        # Append-only: keeps background-task QThreads referenced for their lifetime so
        # they aren't GC'd mid-run (which would warn / crash) even after a patient swap.
        self._task_workers: List[FunctionWorker] = []
        self._progress_dialog: Optional[QProgressDialog] = None
        self._roi_worker: Optional[FunctionWorker] = None
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
        # RCNN ROI detection is lazy + cached + off the UI thread; when on, the inset
        # shows a center crop instantly and is replaced by the detected box afterwards.
        self.auto_detect_roi_checkbox = QCheckBox("Auto-ROI")
        self.auto_detect_roi_checkbox.setChecked(True)
        self.auto_detect_roi_checkbox.setToolTip("Auto-detect the embryo ROI with the RCNN detector")
        self.show_all_depths_checkbox = QCheckBox("All depths")
        self.show_all_depths_checkbox.setToolTip("Open a window showing every focal depth at once")
        self.show_all_depths_checkbox.stateChanged.connect(self.toggle_all_depths)
        for box in (self.show_roi_checkbox, self.auto_detect_roi_checkbox, self.show_all_depths_checkbox):
            row.addWidget(box)

        row.addWidget(self._vsep())

        # Compact background-task indicators (OCR / predictions / ROI boxes).
        self.task_ocr = TaskDot("OCR")
        self.task_pred = TaskDot("Pred")
        self.task_bbox = TaskDot("ROI")
        for dot in (self.task_ocr, self.task_pred, self.task_bbox):
            row.addWidget(dot)

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

        layout.addSpacing(6)
        depths_title = QLabel("BEST DEPTHS")
        depths_title.setObjectName("RailTitle")
        layout.addWidget(depths_title)
        depths_hint = QLabel("up to 3 · white = current view")
        depths_hint.setStyleSheet(f"color: {C_MUTED}; font-size: 10px;")
        layout.addWidget(depths_hint)

        self.focal_depth_buttons: List[QPushButton] = []
        depth_grid = QGridLayout()
        depth_grid.setSpacing(5)
        for index, depth in enumerate(ORDERED_FOCAL_DEPTH):
            chip = QPushButton(depth)
            chip.setCheckable(True)
            chip.setProperty("chip", "depth")
            chip.setProperty("current", "false")
            chip.setCursor(Qt.PointingHandCursor)
            chip.toggled.connect(self.update_focal_depth_selection)
            self.focal_depth_buttons.append(chip)
            depth_grid.addWidget(chip, index // 2, index % 2)
        layout.addLayout(depth_grid)

        layout.addStretch(1)

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

        tools_menu.addSeparator()
        setup_action = QAction("Setup Models…", self)
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
        for url in event.mimeData().urls():
            directory = url.toLocalFile()
            if os.path.isdir(directory):
                self.load_dataset(directory)
                break

    def open_dataset_dialog(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Open dataset folder")
        if directory:
            self.load_dataset(directory)

    def load_dataset(self, directory: str) -> None:
        try:
            self.dataset = DataSet(directory)
            self.update_patient_list()
            self.ensure_model_ready()
            self.auto_select_first_patient()
            QMessageBox.information(self, "Success", f"Loaded dataset from {directory}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load dataset from {directory}\n{exc}")

    def update_patient_list(self) -> None:
        self.patient_combo.blockSignals(True)
        self.patient_combo.clear()
        if self.dataset:
            for patient_ts in self.dataset.get_patient_series():
                self.patient_combo.addItem(os.path.basename(patient_ts.directory))
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

        self.current_patient_ts = series[index]
        self.current_timepoint = 0
        self.current_depth = self.current_patient_ts.default_depth_index()

        # Reset per-patient assisted-labeling state.
        self._pred_arrays = None
        self._ocr_arrays = None
        self._anomaly_cache = None
        self._stage_start = None
        self._stage_end = None
        self._pending_roi = None

        self.update_nav_indicators()
        self.display_image()

        for widget in self._timeline_widgets():
            widget.set_patient(self.current_patient_ts)

        # Auto-compute OCR / predictions / ROI boxes in the background (stale or
        # missing results are recomputed; already-current ones are skipped).
        self.start_background_tasks(self.current_patient_ts)

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
                self.update_focal_depth_checkboxes()
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
            self.update_focal_depth_checkboxes()
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

        Uses the RCNN box cached for this (depth, timepoint) when available, otherwise
        an instant center crop. Detection itself runs off the UI thread (see
        ``_schedule_roi_detection``) so scrolling is never blocked.
        """
        patient = self.current_patient_ts
        if patient is None:
            return rcnn_roi.center_crop(image)

        depth_name = patient.get_depth_name(self.current_depth)
        roi, is_cached = rcnn_roi.roi_for_display(
            image, patient, depth_name, self.current_timepoint
        )
        # Skip per-frame detection while the batch bbox task is running — it writes the
        # same rcnn_boxes cache, and two writer threads would race.
        bbox_task_running = self.task_bbox.worker is not None
        if (
            not is_cached
            and self.auto_detect_roi_checkbox.isChecked()
            and not self._roi_detect_failed
            and not bbox_task_running
        ):
            self._schedule_roi_detection(image, depth_name, self.current_timepoint)
        return roi

    # ------------------------------------------------------------------
    # Background RCNN ROI detection for the current frame
    # ------------------------------------------------------------------
    def _schedule_roi_detection(self, image: np.ndarray, depth_name: str, timepoint: int) -> None:
        """Detect+cache the box for one frame off-thread; only the latest is kept."""
        request = (image.copy(), depth_name, timepoint)
        if self._roi_worker is not None and self._roi_worker.isRunning():
            self._pending_roi = request
            return
        self._start_roi_worker(request)

    def _start_roi_worker(self, request) -> None:
        image, depth_name, timepoint = request
        patient = self.current_patient_ts
        if patient is None:
            return

        def job(progress_cb, message_cb, should_cancel):
            return rcnn_roi.detect_and_cache_frame(image, patient, depth_name, timepoint)

        worker = FunctionWorker(job, self)
        self._roi_worker = worker
        worker.succeeded.connect(lambda _result, d=depth_name, t=timepoint: self._on_roi_detected(d, t))
        worker.failed.connect(self._on_roi_detect_failed)
        worker.finished.connect(self._on_roi_worker_finished)
        worker.start()

    def _on_roi_detected(self, depth_name: str, timepoint: int) -> None:
        # Refresh the ROI view only if the user is still on the detected frame.
        if (
            self.current_patient_ts is not None
            and self.current_patient_ts.get_depth_name(self.current_depth) == depth_name
            and self.current_timepoint == timepoint
            and self.show_roi_checkbox.isChecked()
        ):
            self.refresh_roi_view()

    def _on_roi_detect_failed(self, error: str) -> None:
        # Detection failed (e.g. weights/torch missing): stop auto-detecting and tell
        # the user once. Labeling continues with center crops.
        self._roi_detect_failed = True
        self.auto_detect_roi_checkbox.setChecked(False)
        QMessageBox.warning(
            self,
            "ROI detection unavailable",
            f"RCNN ROI detection failed and was disabled:\n{error}\n\n"
            "Use Tools > Setup Models to install weights, then re-enable "
            "'Auto-detect ROI'.",
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

    def update_focal_depth_checkboxes(self) -> None:
        if self.current_patient_ts is None:
            return
        self._updating_best_depth_checks = True
        selected_depths = self.current_patient_ts.get_best_depths(self.current_timepoint)
        self.selected_focal_depths = selected_depths
        current_depth_name = self.current_patient_ts.get_depth_name(self.current_depth)
        for chip in self.focal_depth_buttons:
            chip.setChecked(chip.text() in selected_depths)
            self._set_chip_property(chip, "current", chip.text() == current_depth_name)
        self._updating_best_depth_checks = False

    def update_focal_depth_selection(self) -> None:
        if self._updating_best_depth_checks or self.current_patient_ts is None:
            return

        selected_depths = [chip.text() for chip in self.focal_depth_buttons if chip.isChecked()]
        if len(selected_depths) > 3:
            sender = self.sender()
            if isinstance(sender, QPushButton):
                sender.blockSignals(True)
                sender.setChecked(False)
                sender.blockSignals(False)
            selected_depths = [chip.text() for chip in self.focal_depth_buttons if chip.isChecked()]
            QMessageBox.warning(self, "Too many depths", "Select at most three focal depths.")

        self.selected_focal_depths = selected_depths
        self.current_patient_ts.set_best_depths(self.current_timepoint, selected_depths)

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
        self.refresh_aws_state()
        local = setup_models.local_models()
        fetch_label = "⟳ Fetch model list from S3…"
        items = local + [fetch_label]
        choice, ok = QInputDialog.getItem(
            self,
            "Setup Models",
            "Choose a local model, or fetch the list from S3:",
            items,
            0,
            False,
        )
        if not ok:
            return
        if choice == fetch_label:
            if not self._require_aws_or_warn():
                return
            self._fetch_and_choose_s3_model()
            return
        self._select_model(choice)

    def _fetch_and_choose_s3_model(self) -> None:
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
                self._select_model(choice)

        self._run_worker("Fetch S3 model list", job, on_success, determinate=False)

    def _select_model(self, model_name: str) -> None:
        missing = setup_models.missing_files(model_name)
        if not missing:
            self.select_model(model_name)
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
        self._download_and_select(model_name)

    def _download_and_select(self, model_name: str) -> None:
        """Download ``model_name`` (+ rcnn.pt) from S3, then make it the active model."""
        def job(progress_cb, message_cb, should_cancel):
            return setup_models.download_model(
                model_name, progress_cb=message_cb, should_cancel=should_cancel
            )

        def on_success(_downloaded):
            self.select_model(model_name)
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
            return predmod.compute_and_save_predictions(
                patient, model_name, progress_cb=progress_cb, should_cancel=should_cancel
            )

        self._launch_task(self.task_pred, patient, job)

    # ==================================================================
    # Background tasks: OCR / predictions / ROI bbox (Phases 2, 3, 4)
    #
    # Each runs off the UI thread on its own progress bar. They are auto-started on
    # patient load and can also be (re-)triggered from the Tools menu.
    # ==================================================================
    def start_background_tasks(self, patient) -> None:
        """Auto-start OCR / predictions / bbox for ``patient`` (skipping current ones).

        Any tasks still running for a previous patient are cancelled first.
        """
        for task_widget in (self.task_ocr, self.task_pred, self.task_bbox):
            task_widget.cancel()

        if patient is None or patient.num_timepoints() == 0:
            for task_widget in (self.task_ocr, self.task_pred, self.task_bbox):
                task_widget.set_idle()
            return

        # OCR
        if ocr_module.has_ocr_times(patient) and not ocr_module.ocr_stale(patient):
            self.task_ocr.set_done("cached")
        else:
            self._launch_task(
                self.task_ocr,
                patient,
                lambda p, m, c: ocr_module.compute_and_save_ocr_times(patient, progress_cb=p, should_cancel=c),
            )

        # Predictions. Cached-and-current wins (no model needed); only flag a missing
        # model when we'd actually have to compute.
        model = self.selected_model
        if patient.has_predictions() and not predmod.predictions_stale(patient):
            meta = patient.load_prediction_metadata()
            self.task_pred.set_done("cached", tooltip=f"cached from {meta.get('model_name', '?')}")
        elif model and setup_models.missing_files(model):
            self.task_pred.set_error(
                "weights missing",
                tooltip=f"'{model}' weights not found — use Tools ▸ Setup Models to download or pick a model.",
            )
        elif model:
            self._launch_task(
                self.task_pred,
                patient,
                lambda p, m, c: predmod.compute_and_save_predictions(patient, model, progress_cb=p, should_cancel=c),
            )
        else:
            self.task_pred.set_idle("no model")

        # ROI bounding boxes for every populated depth (self-skips already-cached frames).
        depths = [patient.get_depth_name(i) for i in patient.populated_depth_indices()]
        self._launch_task(
            self.task_bbox,
            patient,
            lambda p, m, c: rcnn_roi.detect_boxes_for_patient(patient, depths=depths, progress_cb=p, should_cancel=c),
            on_success=lambda _result: self.refresh_roi_view(),
        )

    def _launch_task(self, task_widget, patient, job_fn, on_success=None) -> None:
        worker = FunctionWorker(job_fn, self)
        self._task_workers.append(worker)  # keep referenced for the worker's lifetime

        def succeeded(result):
            # Ignore results for a patient the user has since navigated away from.
            if self.current_patient_ts is patient:
                self._invalidate_assist_caches()
                self.update_prediction_label()
                self.update_label_buttons()
                for widget in self._timeline_widgets():
                    widget.refresh()
                if on_success is not None:
                    on_success(result)

        task_widget.start(worker, on_success=succeeded)

    def run_ocr(self) -> None:
        if self.current_patient_ts is None:
            QMessageBox.warning(self, "No patient selected", "Select a patient first.")
            return
        patient = self.current_patient_ts
        self._launch_task(
            self.task_ocr,
            patient,
            lambda p, m, c: ocr_module.compute_and_save_ocr_times(patient, progress_cb=p, should_cancel=c),
        )

    def run_predictions(self, force: bool = False) -> None:
        if self.current_patient_ts is None:
            QMessageBox.warning(self, "No patient selected", "Select a patient first.")
            return
        patient = self.current_patient_ts

        if not force and patient.has_predictions() and not predmod.predictions_stale(patient):
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
        self._launch_task(
            self.task_pred,
            patient,
            lambda p, m, c: predmod.compute_and_save_predictions(patient, model_name, progress_cb=p, should_cancel=c),
        )

    def detect_rois(self) -> None:
        if self.current_patient_ts is None:
            QMessageBox.warning(self, "No patient selected", "Select a patient first.")
            return
        patient = self.current_patient_ts

        populated = [patient.get_depth_name(i) for i in patient.populated_depth_indices()]
        options = {
            "Current depth only": [patient.get_depth_name(self.current_depth)],
            "Inference depths (F-15, F0, F15)": list(predmod.eb.INFERENCE_DEPTHS),
            "Populated depths": populated,
            "All depths": list(patient.depths),
        }
        choice, ok = QInputDialog.getItem(
            self, "Detect ROIs", "Detect RCNN boxes for:", list(options.keys()), 2, False
        )
        if not ok:
            return
        depths = options[choice]
        self._roi_detect_failed = False
        self._launch_task(
            self.task_bbox,
            patient,
            lambda p, m, c: rcnn_roi.detect_boxes_for_patient(patient, depths=depths, progress_cb=p, should_cancel=c),
            on_success=lambda _result: self.refresh_roi_view(),
        )

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
        total_best_depth_complete = 0

        for patient in self.dataset.get_patient_series():
            n_timepoints = patient.num_timepoints()
            timepoint_labels = patient.labels.get("timepoint_labels", {})
            best_depths = patient.labels.get("best_depths", {})

            n_tp = len(timepoint_labels)
            n_best = sum(1 for value in best_depths.values() if isinstance(value, list) and len(value) == 3)

            total_timepoints += n_timepoints
            total_timepoint_labeled += n_tp
            total_best_depth_complete += n_best

            for label in timepoint_labels.values():
                class_counts[label] += 1

            for event in patient.label_metadata:
                events.append(event)

            patient_rows.append(
                {
                    "patient_id": patient.patient_id,
                    "num_timepoints": n_timepoints,
                    "timepoint_pct": 100.0 * n_tp / n_timepoints if n_timepoints else 0.0,
                    "best_depth_pct": 100.0 * n_best / n_timepoints if n_timepoints else 0.0,
                }
            )

        overall_timepoint_pct = 100.0 * total_timepoint_labeled / total_timepoints if total_timepoints else 0.0
        overall_best_depth_pct = 100.0 * total_best_depth_complete / total_timepoints if total_timepoints else 0.0

        summary_text = (
            f"Dataset: {self.dataset.root_directory}\n"
            f"Patients: {self.dataset.num_patients()} | Timepoints: {total_timepoints}\n"
            f"Timepoint labels complete: {total_timepoint_labeled}/{total_timepoints} "
            f"({overall_timepoint_pct:.1f}%)\n"
            f"Best-depth labels complete: {total_best_depth_complete}/{total_timepoints} "
            f"({overall_best_depth_pct:.1f}%)"
        )

        return {
            "patient_rows": patient_rows,
            "class_counts": class_counts,
            "events": events,
            "summary_text": summary_text,
            "overall_timepoint_pct": overall_timepoint_pct,
            "overall_best_depth_pct": overall_best_depth_pct,
        }

    def _make_progress_bar_chart(self, summary: Dict):
        labels = ["OVERALL"] + [row["patient_id"] for row in summary["patient_rows"]]
        timepoint_values = [summary["overall_timepoint_pct"]] + [row["timepoint_pct"] for row in summary["patient_rows"]]
        best_depth_values = [summary["overall_best_depth_pct"]] + [row["best_depth_pct"] for row in summary["patient_rows"]]

        x = np.arange(len(labels))
        width = 0.38

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - width / 2, timepoint_values, width, label="Timepoint label")
        ax.bar(x + width / 2, best_depth_values, width, label="Best depths")
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
            if event.get("event_type") == "best_depths" and not event.get("complete", False):
                continue
            timestamp = event.get("timestamp")
            if not timestamp:
                continue
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                continue
            created_events.append((dt.date(), event.get("event_type")))

        fig, ax = plt.subplots(figsize=(12, 4))
        if not created_events:
            ax.set_title("Labeling Progress Over Time")
            ax.text(0.5, 0.5, "No label metadata yet", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            fig.tight_layout()
            return fig

        first_day = min(day for day, _ in created_events)
        daily_counts = defaultdict(lambda: {"timepoint_label": 0, "best_depths": 0})
        for day, event_type in created_events:
            daily_counts[(day - first_day).days][event_type] += 1

        max_day = max(daily_counts.keys())
        xs = list(range(max_day + 1))
        tp_cumulative = []
        bd_cumulative = []
        total_cumulative = []
        tp_running = 0
        bd_running = 0
        for day_index in xs:
            tp_running += daily_counts[day_index]["timepoint_label"]
            bd_running += daily_counts[day_index]["best_depths"]
            tp_cumulative.append(tp_running)
            bd_cumulative.append(bd_running)
            total_cumulative.append(tp_running + bd_running)

        ax.plot(xs, tp_cumulative, marker="o", label="Timepoint labels")
        ax.plot(xs, bd_cumulative, marker="o", label="Best-depth labels")
        ax.plot(xs, total_cumulative, marker="o", label="Aggregate")
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
