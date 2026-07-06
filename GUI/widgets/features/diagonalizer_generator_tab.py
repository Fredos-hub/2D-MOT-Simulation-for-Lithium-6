"""Offline diagonalizer-table generator tab (phase 04-05).

A general table generator (D-05): per-manifold ``(I, J, g_J, A_hfs, B_hfs)``
inputs plus a ``|B|`` range and node count drive an offline
``DiagonalizerWorker`` that writes an NPZ — mirroring the Sample Generator +
``OvenWorker`` pattern. The generated table is consumed by
``Lithium6DiagonalizerTableInteraction`` via the Simulation
``interaction_table_file`` config key (see the Simulation settings tab).

``generate_table`` currently reproduces the fixed Li-6 D2 line, so the
manifold fields are pre-filled with the Li-6 D2 constants; general species/line
support is deferred and the worker validates the fields against Li-6 D2.
"""
from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from PyQt5.QtCore import QDir, QLocale, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import src.diagonalizer_setup as ds
from GUI.workers.diagonalizer_worker import DiagonalizerWorker

DEFAULT_OUTPUT = "interaction_tables/li6_d2_table.npz"
POLARIZATIONS = ("σ⁻ (q=0)", "π (q=1)", "σ⁺ (q=2)")


class DiagonalizerGeneratorTab(QWidget):
    tableCreated = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.worker = None  # active DiagonalizerWorker, if any
        self._init_ui()

    # ---- UI construction ------------------------------------------------ #
    def _init_ui(self) -> None:
        main = QVBoxLayout(self)

        upper = QHBoxLayout()
        upper.addWidget(self._build_settings_box(), 1)
        upper.addWidget(self._build_preview_box(), 2)
        main.addLayout(upper)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main.addWidget(self.progress_bar)

        self.status_label = QLabel("Status: Not started")
        main.addWidget(self.status_label)

    def _build_settings_box(self) -> QGroupBox:
        group = QGroupBox("Diagonalizer Table Settings")
        layout = QFormLayout(group)

        note = QLabel(
            "Generates the Li-6 D2 table. The manifold constants below are "
            "shown for the general tool (D-05); species generalization is "
            "deferred, so they are validated against Li-6 D2 for now."
        )
        note.setWordWrap(True)
        layout.addRow(note)

        # Shared nuclear spin
        self.i_spin = self._dspin(ds.LI6_D2["I"], 0.0, 10.0, 1)
        layout.addRow("Nuclear spin I:", self.i_spin)

        # Ground manifold
        g = ds.LI6_D2["ground"]
        self.g_j = self._dspin(g["J"], 0.0, 10.0, 1)
        self.g_gj = self._dspin(g["g_J"], -10.0, 10.0, 3)
        self.g_a = self._dspin(g["A"] / 1e6, -1e4, 1e4, 3, " MHz")
        self.g_b = self._dspin(g["B_hfs"] / 1e6, -1e4, 1e4, 3, " MHz")
        layout.addRow("Ground J:", self.g_j)
        layout.addRow("Ground g_J:", self.g_gj)
        layout.addRow("Ground A_hfs:", self.g_a)
        layout.addRow("Ground B_hfs:", self.g_b)

        # Excited manifold
        e = ds.LI6_D2["excited"]
        self.e_j = self._dspin(e["J"], 0.0, 10.0, 1)
        self.e_gj = self._dspin(e["g_J"], -10.0, 10.0, 3)
        self.e_a = self._dspin(e["A"] / 1e6, -1e4, 1e4, 3, " MHz")
        self.e_b = self._dspin(e["B_hfs"] / 1e6, -1e4, 1e4, 3, " MHz")
        layout.addRow("Excited J:", self.e_j)
        layout.addRow("Excited g_J:", self.e_gj)
        layout.addRow("Excited A_hfs:", self.e_a)
        layout.addRow("Excited B_hfs:", self.e_b)

        # |B| range + node count
        self.b_min = self._dspin(0.0, 0.0, 100.0, 4, " T")
        self.b_max = self._dspin(1.0, 0.0, 100.0, 4, " T")
        layout.addRow("|B| range:", self._hbox(self.b_min, self.b_max))

        self.node_spin = QSpinBox()
        self.node_spin.setRange(2, 100_000)
        self.node_spin.setValue(200)
        layout.addRow("Node count:", self.node_spin)

        # Output path
        self.output_line = QLineEdit(DEFAULT_OUTPUT)
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse_output)
        layout.addRow(
            "Output file:", self._hbox(self.output_line, self.browse_btn)
        )

        # Generate / Cancel
        self.generate_btn = QPushButton("Generate Table")
        self.generate_btn.clicked.connect(self._generate)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setEnabled(False)
        layout.addRow("", self._hbox(self.generate_btn, self.cancel_btn))

        return group

    def _build_preview_box(self) -> QGroupBox:
        box = QGroupBox("Transition preview")
        layout = QVBoxLayout(box)

        # Transition selectors: ground state, excited state, polarization.
        sel = QHBoxLayout()
        self.gs_combo = QComboBox()
        self.gs_combo.addItems([f"GS{i}" for i in range(6)])
        self.gs_combo.setCurrentIndex(5)
        self.es_combo = QComboBox()
        self.es_combo.addItems([f"ES{i}" for i in range(12)])
        self.es_combo.setCurrentIndex(11)
        self.pol_combo = QComboBox()
        self.pol_combo.addItems(POLARIZATIONS)
        self.pol_combo.setCurrentIndex(2)  # σ⁺, the cycling default
        for label, combo in (
            ("Ground:", self.gs_combo),
            ("Excited:", self.es_combo),
            ("Polarization:", self.pol_combo),
        ):
            sel.addWidget(QLabel(label))
            sel.addWidget(combo)
        sel.addStretch(1)
        layout.addLayout(sel)

        self._fig = Figure(figsize=(5, 4), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        # Navigation toolbar gives interactive zoom/pan (and thus arbitrary
        # displayed |B| range) plus a reset-to-full-range home button.
        self._toolbar = NavigationToolbar(self._canvas, box)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self._preview_data = None  # cached NPZ arrays for re-plotting
        for combo in (self.gs_combo, self.es_combo, self.pol_combo):
            combo.currentIndexChanged.connect(self._redraw_preview)

        ax = self._fig.add_subplot(111)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("|B| (T)")
        ax.set_ylabel("Line shift (MHz)")
        self._ax = ax
        return box

    # ---- helpers -------------------------------------------------------- #
    def _dspin(
        self, value, lo, hi, decimals, suffix=""
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setLocale(QLocale(QLocale.C))
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        if suffix:
            spin.setSuffix(suffix)
        spin.setValue(value)
        return spin

    def _hbox(self, *widgets) -> QHBoxLayout:
        h = QHBoxLayout()
        for w in widgets:
            h.addWidget(w)
        return h

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Table", DEFAULT_OUTPUT, "NumPy tables (*.npz)"
        )
        if path:
            self.output_line.setText(QDir().relativeFilePath(path))

    def _params(self) -> dict:
        return {
            "I": self.i_spin.value(),
            "ground_J": self.g_j.value(),
            "ground_g_J": self.g_gj.value(),
            "ground_A": self.g_a.value() * 1e6,
            "ground_B_hfs": self.g_b.value() * 1e6,
            "excited_J": self.e_j.value(),
            "excited_g_J": self.e_gj.value(),
            "excited_A": self.e_a.value() * 1e6,
            "excited_B_hfs": self.e_b.value() * 1e6,
            "b_min": self.b_min.value(),
            "b_max": self.b_max.value(),
            "n_nodes": self.node_spin.value(),
            "output_file": self.output_line.text().strip(),
        }

    # ---- generation lifecycle ------------------------------------------ #
    def _generate(self) -> None:
        self._set_running(True)
        self.status_label.setText("Status: Generating…")
        self.worker = DiagonalizerWorker(self._params())
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.cancelled.connect(self._on_cancelled)
        self.worker.start()

    def _set_running(self, running: bool) -> None:
        self.generate_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)

    def _cancel(self) -> None:
        if self.is_busy():
            self.worker.cancel()
            self.status_label.setText("Status: Cancelling…")

    def is_busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def cancel_and_wait(self) -> None:
        """Request cancellation and block until the worker has finished."""
        if self.is_busy():
            self.worker.cancel()
            self.worker.wait()

    def _on_finished(self, path: str) -> None:
        self._set_running(False)
        self.status_label.setText(f"Status: Done → {path}")
        self._update_preview(path)
        self.tableCreated.emit(path)

    def _on_error(self, message: str) -> None:
        self._set_running(False)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Status: Error — {message}")

    def _on_cancelled(self) -> None:
        self._set_running(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Cancelled.")

    def _update_preview(self, path: str) -> None:
        """Cache the generated NPZ, then plot the selected transition."""
        try:
            with np.load(path) as data:
                self._preview_data = {
                    "b_axis": data["b_axis"],
                    "pos_table": data["pos_table"],
                    "strength_table": data["strength_table"],
                }
        except Exception:
            self._preview_data = None
            return
        self._redraw_preview()

    def _redraw_preview(self, *_) -> None:
        """Plot shift + strength vs |B| for the currently selected line."""
        if not self._preview_data:
            return
        gs = self.gs_combo.currentIndex()
        es = self.es_combo.currentIndex()
        pol = self.pol_combo.currentIndex()
        b_axis = self._preview_data["b_axis"]
        shift = self._preview_data["pos_table"][:, gs, es] / 1e6  # MHz
        strength = self._preview_data["strength_table"][:, gs, es, pol]

        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.plot(b_axis, shift, color="C0", label="line shift")
        ax.set_xlabel("|B| (T)")
        ax.set_ylabel("Line shift (MHz)", color="C0")
        ax.set_title(f"GS{gs} → ES{es}")
        ax.grid(True, alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(b_axis, strength, color="C1", label="strength")
        ax2.set_ylabel(f"{POLARIZATIONS[pol]} strength |CG|²", color="C1")

        self._ax = ax
        self._canvas.draw_idle()
