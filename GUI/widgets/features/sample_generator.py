
from __future__ import annotations

from functools import partial
import csv

import numpy as np
import pandas as pd
import scipy.constants as scc

from PyQt5.QtWidgets import (
    QWidget, QFormLayout, QSpinBox, QDoubleSpinBox, QHBoxLayout, QVBoxLayout,
    QRadioButton, QProgressBar, QStackedWidget, QLabel, QPushButton,
    QListWidget, QFileDialog, QLineEdit, QGroupBox, QComboBox
)
from PyQt5.QtCore import pyqtSignal, Qt, QLocale, QDir

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from GUI.workers.oven_worker import OvenWorker


class SampleGeneratorTab(QWidget):
    # Signals emitted when a sample is generated
    sampleCreated = pyqtSignal(str)

    REQUIRED_FILE_COLUMNS = {
        "atom_id",
        "subjective_time",
        "position_x", "position_y", "position_z",
        "velocity_x", "velocity_y", "velocity_z",
        "excitation_count",
        "current_groundstate",
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        # File-based state
        self.file_df: pd.DataFrame | None = None
        self.step_times_s: np.ndarray = np.array([], dtype=float)
        self.step_summary: pd.DataFrame = pd.DataFrame()
        self.worker = None   # active OvenWorker, if any

        # Plot widgets are created in _init_ui()
        self._init_ui()

    def _init_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)

        upper_layout = QHBoxLayout()
        lower_layout = QVBoxLayout()

        # Left side: option selection and parameter editing
        options_layout = QVBoxLayout()
        options_layout.addWidget(QLabel("Sample Creation Mode:"))

        self.oven_radio = QRadioButton("Create Sample from Oven")
        self.file_radio = QRadioButton("Create Sample from File")
        self.oven_radio.setChecked(True)
        self.oven_radio.toggled.connect(self._on_mode_changed)

        options_layout.addWidget(self.oven_radio)
        options_layout.addWidget(self.file_radio)

        # Stacked widget for mode-specific controls
        self.stacked = QStackedWidget()
        self.stacked.addWidget(self._build_oven_ui())
        self.stacked.addWidget(self._build_file_ui())

        options_layout.addWidget(self.stacked)
        options_layout.addStretch()

        upper_layout.addLayout(options_layout, 1)

        # Right side: interactive plots
        self.graph_stack = QStackedWidget()
        self.file_graph_widget = self._build_file_graph_ui()
        self.oven_graph_widget = self._build_oven_graph_ui()
        self.graph_stack.addWidget(self.file_graph_widget)
        self.graph_stack.addWidget(self.oven_graph_widget)
        self.graph_stack.setCurrentWidget(self.oven_graph_widget)

        upper_layout.addWidget(self.graph_stack, 2)

        # Progress and status
        self.dist_progress_bar = QProgressBar()
        self.dist_progress_bar.setValue(0)
        lower_layout.addWidget(self.dist_progress_bar)

        self.statusLabel = QLabel("Status: Not started")
        lower_layout.addWidget(self.statusLabel)

        main_layout.addLayout(upper_layout)
        main_layout.addLayout(lower_layout)

        self._connect_oven_preview_signals()

        # Make sure the UI reflects the current state.
        self._refresh_file_ui_state()
        self._on_mode_changed(False)

    def _build_canvas_box(self, title: str, prefix: str) -> QGroupBox:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)

        fig = Figure(figsize=(5, 3), tight_layout=True)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)

        ax = fig.add_subplot(111)
        ax.grid(True, alpha=0.25)

        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        setattr(self, f"{prefix}_fig", fig)
        setattr(self, f"{prefix}_canvas", canvas)
        setattr(self, f"{prefix}_ax", ax)

        return box

    def _build_file_graph_ui(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.position_plot_box = self._build_canvas_box("Average position by step", "position")
        self.velocity_plot_box = self._build_canvas_box("Average velocity by step", "velocity")
        layout.addWidget(self.position_plot_box)
        layout.addWidget(self.velocity_plot_box)

        return widget

    def _build_oven_graph_ui(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.distribution_plot_box = self._build_canvas_box(
            "Maxwell-Boltzmann preview", "distribution"
        )
        self.oven_draft_plot_box = self._build_canvas_box(
            "2D oven draft", "oven_draft"
        )

        layout.addWidget(self.distribution_plot_box, 1)
        layout.addWidget(self.oven_draft_plot_box, 1)

        return widget

    def _build_oven_ui(self):
        group = QGroupBox("Oven Sample Settings")
        layout = QFormLayout()
        self.aperture_data = []

        # Atom Mass (default 6.015 u, decimal ".")
        self.atom_mass_spin = QDoubleSpinBox()
        self.atom_mass_spin.setRange(0, 500)
        self.atom_mass_spin.setSuffix(" u")
        self.atom_mass_spin.setLocale(QLocale(QLocale.C))
        self.atom_mass_spin.setDecimals(3)
        self.atom_mass_spin.setValue(6.015)
        layout.addRow("Atomic Mass:", self.atom_mass_spin)

        # Boltzmann Distribution
        self.distribution_combo = QComboBox()
        self.distribution_combo.addItems([
            "Maxwell-Boltzmann-Distribution v2",
            "Maxwell-Boltzmann-Distribution v3"
        ])
        layout.addRow("Distribution:", self.distribution_combo)

        # Number of atoms
        self.number_of_atoms_spin = QSpinBox()
        self.number_of_atoms_spin.setRange(0, 100_000_000)
        self.number_of_atoms_spin.setValue(100_000)
        layout.addRow("Number of sample Atoms to be generated:", self.number_of_atoms_spin)

        # Temperature
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0, 5000)
        self.temp_spin.setSuffix(" K")
        self.temp_spin.setLocale(QLocale(QLocale.C))
        self.temp_spin.setValue(743)
        layout.addRow("Temperature:", self.temp_spin)

        # Velocity range
        self.vel_min = QDoubleSpinBox()
        self.vel_max = QDoubleSpinBox()
        for spin in (self.vel_min, self.vel_max):
            spin.setSuffix(" m/s")
            spin.setRange(0, 10000)
            spin.setLocale(QLocale(QLocale.C))
            spin.setDecimals(1)

        self.vel_max.setValue(150)
        layout.addRow("Velocity range:", self._hbox(self.vel_min, self.vel_max))

        # Oven geometry: radius and y-position
        self.oven_radius_spin = QDoubleSpinBox()
        self.oven_radius_spin.setRange(0, 1e3)
        self.oven_radius_spin.setSuffix(" mm")
        self.oven_radius_spin.setLocale(QLocale(QLocale.C))
        self.oven_radius_spin.setValue(25)
        self.oven_radius_spin.setDecimals(1)

        self.oven_ypos_spin = QDoubleSpinBox()
        self.oven_ypos_spin.setRange(-1e3, 1e3)
        self.oven_ypos_spin.setSuffix(" mm")
        self.oven_ypos_spin.setLocale(QLocale(QLocale.C))
        self.oven_ypos_spin.setDecimals(1)

        layout.addRow("Oven radius & y-pos:", self._hbox(self.oven_radius_spin, self.oven_ypos_spin))

        # Aperture geometry
        self.aperture_list = QListWidget()

        self.ap_radius_spin = QDoubleSpinBox()
        self.ap_radius_spin.setRange(0, 1e3)
        self.ap_radius_spin.setSuffix(" mm")
        self.ap_radius_spin.setLocale(QLocale(QLocale.C))
        self.ap_radius_spin.setDecimals(1)

        self.ap_ypos_spin = QDoubleSpinBox()
        self.ap_ypos_spin.setRange(-1e3, 1e3)
        self.ap_ypos_spin.setSuffix(" mm")
        self.ap_ypos_spin.setLocale(QLocale(QLocale.C))
        self.ap_ypos_spin.setDecimals(1)

        self.add_aperture_btn = QPushButton("Add Aperture")
        self.add_aperture_btn.clicked.connect(self._add_aperture)
        self.remove_aperture_btn = QPushButton("Remove Selected")
        self.remove_aperture_btn.clicked.connect(self._remove_aperture)

        layout.addRow("Apertures:", self.aperture_list)
        layout.addRow("Radius & y-pos:", self._hbox(self.ap_radius_spin, self.ap_ypos_spin))
        layout.addRow("", self._hbox(self.add_aperture_btn, self.remove_aperture_btn))

        # Output file
        self.output_line = QLineEdit()
        self.browse_output_btn = QPushButton("Browse…")
        self.browse_output_btn.clicked.connect(lambda: self._browse_output(self.output_line))
        layout.addRow("Output file:", self._hbox(self.output_line, self.browse_output_btn))

        # Generate / Cancel buttons
        self.generate_oven_btn = QPushButton("Generate Sample")
        self.generate_oven_btn.clicked.connect(self._generate_from_oven)
        self.cancel_oven_btn = QPushButton("Cancel")
        self.cancel_oven_btn.clicked.connect(self._cancel_oven)
        self.cancel_oven_btn.setEnabled(False)
        layout.addRow("", self._hbox(self.generate_oven_btn, self.cancel_oven_btn))

        group.setLayout(layout)
        return group

    def _build_file_ui(self):
        group = QGroupBox("File-based Sample Settings")
        layout = QFormLayout()

        # Input file selection
        self.input_line = QLineEdit()
        self.browse_input_btn = QPushButton("Browse…")
        self.browse_input_btn.clicked.connect(self._browse_input)
        layout.addRow("Input file:", self._hbox(self.input_line, self.browse_input_btn))

        # Step selector
        self.step_spin = QSpinBox()
        self.step_spin.setRange(0, 0)
        self.step_spin.valueChanged.connect(self._on_step_changed)
        self.step_time_label = QLabel("0.000 ms")
        self.step_time_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addRow("Step:", self._hbox(self.step_spin, self.step_time_label))

        # Time selector readout
        self.step_info_label = QLabel("Load a file to enable step navigation.")
        layout.addRow("Snapshot:", self.step_info_label)

        # Output file
        self.output_line_2 = QLineEdit()
        self.browse_output_btn_2 = QPushButton("Browse…")
        self.browse_output_btn_2.clicked.connect(lambda: self._browse_output(self.output_line_2))
        layout.addRow("Output file:", self._hbox(self.output_line_2, self.browse_output_btn_2))

        # Generate button
        self.generate_file_btn = QPushButton("Generate Sample")
        self.generate_file_btn.clicked.connect(self._generate_from_file)
        layout.addRow("", self.generate_file_btn)

        group.setLayout(layout)
        return group

    def _connect_oven_preview_signals(self):
        for spin in (
            self.atom_mass_spin,
            self.temp_spin,
            self.vel_min,
            self.vel_max,
            self.oven_radius_spin,
            self.oven_ypos_spin,
        ):
            spin.valueChanged.connect(lambda *_: self._refresh_oven_preview())

        self.distribution_combo.currentIndexChanged.connect(
            lambda *_: self._refresh_oven_preview()
        )

    def _refresh_oven_preview(self):
        if not self.oven_radio.isChecked():
            return

        self._update_oven_distribution_plot()
        self._update_oven_draft_plot()

    def _maxwell_boltzmann_speed_pdf(self, speed: np.ndarray, temperature_k: float, mass_u: float) -> np.ndarray:
        """Maxwell-Boltzmann speed distribution for a monatomic gas."""
        temperature_k = max(float(temperature_k), 1e-12)
        mass_kg = max(float(mass_u), 1e-12) * scc.u

        prefactor = 4.0 * np.pi * (mass_kg / (2.0 * np.pi * scc.k * temperature_k)) ** 1.5
        return prefactor * speed**2 * np.exp(-(mass_kg * speed**2) / (2.0 * scc.k * temperature_k))

    def _update_oven_distribution_plot(self):
        ax = getattr(self, "distribution_ax", None)
        canvas = getattr(self, "distribution_canvas", None)
        if ax is None or canvas is None:
            return

        ax.clear()

        temperature = max(self.temp_spin.value(), 1e-12)
        mass_u = max(self.atom_mass_spin.value(), 1e-12)
        vmin = min(self.vel_min.value(), self.vel_max.value())
        vmax = max(self.vel_min.value(), self.vel_max.value())

        mass_kg = mass_u * scc.u
        v_peak = np.sqrt((2.0 * scc.k * temperature) / mass_kg) if mass_kg > 0 else max(vmax, 1.0)

        domain_max = max(vmax * 2.0, v_peak * 3.5, 1.0)
        speed = np.linspace(0.0, domain_max, 1000)
        pdf = self._maxwell_boltzmann_speed_pdf(speed, temperature, mass_u)

        ax.plot(speed, pdf, label="Full distribution", linewidth=1.8)

        sample_mask = (speed >= vmin) & (speed <= vmax)
        if np.any(sample_mask):
            ax.plot(speed[sample_mask], pdf[sample_mask], label="Sampled clamp", linewidth=2.6)
            ax.fill_between(speed[sample_mask], pdf[sample_mask], alpha=0.18)

        ax.axvline(vmin, linestyle="--", linewidth=1.0)
        ax.axvline(vmax, linestyle="--", linewidth=1.0)

        selected_distribution = self.distribution_combo.currentText()
        ax.set_title(
            f"{selected_distribution} — T={temperature:.1f} K, range {vmin:.1f}…{vmax:.1f} m/s"
        )
        ax.set_xlabel("Speed (m/s)")
        ax.set_ylabel("Probability density")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        canvas.draw_idle()

    def _get_oven_components(self):
        components = [
            {
                "kind": "oven",
                "label": "Oven",
                "y": float(self.oven_ypos_spin.value()),
                "r": float(self.oven_radius_spin.value()),
            }
        ]

        for index, (radius, ypos) in enumerate(self.aperture_data, start=1):
            components.append(
                {
                    "kind": "aperture",
                    "label": f"Aperture {index}",
                    "y": float(ypos),
                    "r": float(radius),
                }
            )

        return sorted(components, key=lambda item: item["y"])

    def _update_oven_draft_plot(self):
        ax = getattr(self, "oven_draft_ax", None)
        canvas = getattr(self, "oven_draft_canvas", None)
        if ax is None or canvas is None:
            return

        ax.clear()

        components = self._get_oven_components()
        if not components:
            ax.set_title("2D oven draft")
            canvas.draw_idle()
            return

        y_values = np.array([c["y"] for c in components], dtype=float)
        radii = np.array([max(c["r"], 0.0) for c in components], dtype=float)
        left_edges = -radii
        right_edges = radii

        # Light hatch for the reachable tube; no solid shading of the oven body.
        if len(components) > 1:
            ax.fill_betweenx(
                y_values,
                left_edges,
                right_edges,
                facecolor="none",
                edgecolor="black",
                hatch="///",
                linewidth=0.0,
                zorder=0,
            )

        # Connect the outlines so the cut-through reads as a continuous schematic.
        if len(components) > 1:
            for idx in range(len(components) - 1):
                ax.plot(
                    [left_edges[idx], left_edges[idx + 1]],
                    [y_values[idx], y_values[idx + 1]],
                    linestyle="--",
                    linewidth=1.0,
                    color="black",
                    alpha=0.85,
                    zorder=1,
                )
                ax.plot(
                    [right_edges[idx], right_edges[idx + 1]],
                    [y_values[idx], y_values[idx + 1]],
                    linestyle="--",
                    linewidth=1.0,
                    color="black",
                    alpha=0.85,
                    zorder=1,
                )

        # Draw the oven and apertures as black outlines.
        for comp in components:
            line_width = 3.0 if comp["kind"] == "oven" else 2.0
            ax.plot(
                [-comp["r"], comp["r"]],
                [comp["y"], comp["y"]],
                linewidth=line_width,
                color="black",
                solid_capstyle="round",
                zorder=2,
            )
            ax.plot(
                [-comp["r"], comp["r"]],
                [comp["y"], comp["y"]],
                marker="o",
                linestyle="None",
                markersize=3,
                color="black",
                zorder=3,
            )

            label = (
                f"Oven: R={comp['r']:.1f} mm, y={comp['y']:.1f} mm"
                if comp["kind"] == "oven"
                else f"{comp['label']}: R={comp['r']:.1f} mm, y={comp['y']:.1f} mm"
            )
            ax.annotate(
                label,
                xy=(comp["r"], comp["y"]),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color="black",
            )

        max_r = max(1.0, float(np.max(radii)) if radii.size else 1.0)
        y_min = float(np.min(y_values)) - max(10.0, max_r * 0.35)
        y_max = float(np.max(y_values)) + max(10.0, max_r * 0.35)

        ax.axvline(0.0, linestyle=":", linewidth=1.0, color="black", alpha=0.55)
        ax.set_xlim(-max_r * 1.35, max_r * 1.35)
        ax.set_ylim(y_min, y_max)
        ax.set_title("2D oven draft (schematic cylindrical cut)")
        ax.set_xlabel("Radial extent x (mm)")
        ax.set_ylabel("Position y (mm)")
        ax.grid(True, alpha=0.25)
        canvas.draw_idle()

    def _hbox(self, *widgets):
        h = QHBoxLayout()
        for w in widgets:
            h.addWidget(w)
        return h

    def _on_mode_changed(self, checked):
        idx = 0 if self.oven_radio.isChecked() else 1
        self.stacked.setCurrentIndex(idx)

        if self.oven_radio.isChecked():
            self.graph_stack.setCurrentWidget(self.oven_graph_widget)
            self._refresh_oven_preview()
        else:
            self.graph_stack.setCurrentWidget(self.file_graph_widget)
            if self.step_times_s.size:
                self._update_step_preview(self.step_spin.value())
            else:
                self._clear_file_plots()

    def _refresh_file_ui_state(self):
        has_file = self.file_df is not None and not self.step_summary.empty
        self.step_spin.setEnabled(has_file)
        self.generate_file_btn.setEnabled(has_file)
        self.step_time_label.setEnabled(has_file)
        self.step_info_label.setEnabled(has_file)

        if has_file:
            max_step = max(0, len(self.step_times_s) - 1)
            self.step_spin.setMaximum(max_step)
            self.step_spin.setValue(min(self.step_spin.value(), max_step))
            self._update_step_preview(self.step_spin.value())
        else:
            self.step_spin.setMaximum(0)
            self.step_spin.setValue(0)
            self.step_time_label.setText("0.000 ms")
            self.step_info_label.setText("Load a file to enable step navigation.")
            self._clear_file_plots()

    def _add_aperture(self):
        r = self.ap_radius_spin.value()
        y = self.ap_ypos_spin.value()
        self.aperture_data.append((r, y))
        self.aperture_list.addItem(f"Aperture: r={r:.1f} mm, y={y:.1f} mm")
        self._refresh_oven_preview()

    def _remove_aperture(self):
        # Delete from the end so row indices stay valid.
        for item in self.aperture_list.selectedItems():
            row = self.aperture_list.row(item)
            if 0 <= row < len(self.aperture_data):
                self.aperture_data.pop(row)
            self.aperture_list.takeItem(row)
        self._refresh_oven_preview()

    def _browse_output(self, target_line_edit: QLineEdit):
        path, _ = QFileDialog.getSaveFileName(self, "Save Output", filter="CSV files (*.csv)")
        if path:
            rel = QDir().relativeFilePath(path)
            target_line_edit.setText(rel)

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Sample File", filter="CSV files (*.csv)")
        if path:
            rel = QDir().relativeFilePath(path)
            self.input_line.setText(rel)
            self._load_file_data(rel)

    def _load_file_data(self, input_file: str):
        try:
            df = pd.read_csv(input_file)
        except Exception as exc:
            self.file_df = None
            self.step_times_s = np.array([], dtype=float)
            self.step_summary = pd.DataFrame()
            self.statusLabel.setText(f"Status: Failed to read input file ({exc})")
            self._refresh_file_ui_state()
            return

        missing = self.REQUIRED_FILE_COLUMNS - set(df.columns)
        if missing:
            self.file_df = None
            self.step_times_s = np.array([], dtype=float)
            self.step_summary = pd.DataFrame()
            self.statusLabel.setText(f"Status: Missing required columns: {', '.join(sorted(missing))}")
            self._refresh_file_ui_state()
            return

        self.file_df = df.copy()

        # Unique step times in seconds, sorted ascending.
        self.step_times_s = np.array(sorted(self.file_df["subjective_time"].dropna().unique()), dtype=float)

        if self.step_times_s.size == 0:
            self.step_summary = pd.DataFrame()
            self.statusLabel.setText("Status: Input file has no valid subjective_time values.")
            self._refresh_file_ui_state()
            return

        # Build a per-step summary for plotting.
        summary = (
            self.file_df
            .groupby("subjective_time", as_index=False)[
                ["position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z"]
            ]
            .mean()
            .sort_values("subjective_time")
            .reset_index(drop=True)
        )
        summary.insert(0, "step", range(len(summary)))
        summary["time_ms"] = summary["subjective_time"] * 1000.0
        self.step_summary = summary

        self.statusLabel.setText(
            f"Status: Loaded {len(df)} rows, {len(self.step_times_s)} steps"
        )
        self._refresh_file_ui_state()

    def _on_step_changed(self, step: int):
        if self.step_times_s.size == 0:
            self.step_time_label.setText("0.000 ms")
            self.step_info_label.setText("Load a file to enable step navigation.")
            return

        self._update_step_preview(step)

    def _update_step_preview(self, step: int):
        if self.step_times_s.size == 0:
            return

        step = max(0, min(step, len(self.step_times_s) - 1))
        time_s = float(self.step_times_s[step])
        time_ms = time_s * 1000.0

        self.step_time_label.setText(f"{time_ms:.3f} ms")
        self.step_info_label.setText(f"Selected step {step} → {time_ms:.3f} ms")

        self._update_file_plots(step)

    def _clear_file_plots(self):
        for ax in (getattr(self, "position_ax", None), getattr(self, "velocity_ax", None)):
            if ax is not None:
                ax.clear()

        for canvas in (getattr(self, "position_canvas", None), getattr(self, "velocity_canvas", None)):
            if canvas is not None:
                canvas.draw_idle()

    def _update_file_plots(self, selected_step: int):
        if self.step_summary.empty:
            self._clear_file_plots()
            return

        summary = self.step_summary
        x = summary["step"].to_numpy()

        pos_series = [
            ("x", summary["position_x"].to_numpy()),
            ("y", summary["position_y"].to_numpy()),
            ("z", summary["position_z"].to_numpy()),
        ]
        vel_series = [
            ("velocity_x", summary["velocity_x"].to_numpy()),
            ("velocity_y", summary["velocity_y"].to_numpy()),
            ("velocity_z", summary["velocity_z"].to_numpy()),
        ]

        self._draw_series_plot(
            ax=self.position_ax,
            canvas=self.position_canvas,
            x=x,
            series=pos_series,
            selected_step=selected_step,
            title="Average position by step",
            y_label="Position",
            selected_time_ms=float(summary.loc[selected_step, "time_ms"]),
        )

        self._draw_series_plot(
            ax=self.velocity_ax,
            canvas=self.velocity_canvas,
            x=x,
            series=vel_series,
            selected_step=selected_step,
            title="Average velocity by step",
            y_label="Velocity",
            selected_time_ms=float(summary.loc[selected_step, "time_ms"]),
        )

    def _draw_series_plot(
        self,
        ax,
        canvas,
        x: np.ndarray,
        series,
        selected_step: int,
        title: str,
        y_label: str,
        selected_time_ms: float,
    ):
        ax.clear()

        for label, y in series:
            ax.plot(x, y, label=label, linewidth=1.5)

        if len(x) > 0:
            selected_step = max(0, min(selected_step, len(x) - 1))
            ax.axvline(selected_step, linestyle="--", linewidth=1, alpha=0.7)
            for _, y in series:
                ax.plot(selected_step, y[selected_step], marker="o", markersize=5)

        ax.set_title(f"{title} — selected time {selected_time_ms:.3f} ms")
        ax.set_xlabel("Step")
        ax.set_ylabel(y_label)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        canvas.draw_idle()

    def _generate_from_oven(self):
        # 1) read all the UI state into a plain dict
        params = {
            "atom_mass": self.atom_mass_spin.value(),
            "distribution": self.distribution_combo.currentText(),
            "num_atoms": self.number_of_atoms_spin.value(),
            "temperature": self.temp_spin.value(),
            "vmin": self.vel_min.value(),
            "vmax": self.vel_max.value(),
            "oven_geometry": (
                self.oven_radius_spin.value(),
                self.oven_ypos_spin.value(),
            ),
            "apertures": self.aperture_data,
            "output_file": self.output_line.text(),
        }

        # 2) reflect running state in the buttons
        self._set_oven_running(True)

        # 3) spin up worker thread
        self.worker = OvenWorker(params)
        self.worker.progress.connect(self.dist_progress_bar.setValue)
        self.worker.finished.connect(self._on_oven_finished)
        self.worker.error.connect(self._on_oven_error)
        self.worker.cancelled.connect(self._on_oven_cancelled)
        self.worker.start()

        self.statusLabel.setText("Status: Running…")

    def _set_oven_running(self, running: bool):
        self.generate_oven_btn.setEnabled(not running)
        self.cancel_oven_btn.setEnabled(running)

    def _cancel_oven(self):
        if self.is_busy():
            self.worker.cancel()
            self.statusLabel.setText("Status: Cancelling…")

    def is_busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def cancel_and_wait(self):
        """Request cancellation and block until the worker thread has finished."""
        if self.is_busy():
            self.worker.cancel()
            self.worker.wait()

    def _on_oven_finished(self, filename):
        self._set_oven_running(False)
        self.statusLabel.setText(f"Status: Done → {filename}")
        self.sampleCreated.emit(filename)

    def _on_oven_error(self, message):
        self._set_oven_running(False)
        self.dist_progress_bar.setValue(0)
        self.statusLabel.setText(f"Status: Error — {message}")

    def _on_oven_cancelled(self):
        self._set_oven_running(False)
        self.dist_progress_bar.setValue(0)
        self.statusLabel.setText("Status: Cancelled.")

    def _generate_from_file(self):
        input_file = self.input_line.text().strip()
        output_file = self.output_line_2.text().strip()

        if not input_file:
            self.statusLabel.setText("Status: Please choose an input file.")
            return
        if not output_file:
            self.statusLabel.setText("Status: Please choose an output file.")
            return

        # Load or reload the data.
        if self.file_df is None or self.file_df.empty:
            self._load_file_data(input_file)

        if self.file_df is None or self.step_times_s.size == 0:
            self.statusLabel.setText("Status: Could not load the input file.")
            return

        step = int(self.step_spin.value())
        if step < 0 or step >= len(self.step_times_s):
            self.statusLabel.setText("Status: Selected step is out of range.")
            return

        selected_time_s = float(self.step_times_s[step])

        # Use the selected step's exact time slice.
        # The file contains times in seconds; the UI displays ms.
        step_slice = self.file_df[np.isclose(self.file_df["subjective_time"], selected_time_s)]

        if step_slice.empty:
            self.statusLabel.setText(
                f"Status: No rows found for step {step} ({selected_time_s * 1000.0:.3f} ms)."
            )
            return

        # If the file has one row per atom per step, this stays one row per atom.
        # If there are duplicates, keep the last row for each atom at that step.
        snapshot = (
            step_slice.sort_values(["atom_id", "subjective_time"])
            .groupby("atom_id", as_index=False)
            .tail(1)
            .copy()
        )

        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "step",
                "x", "y", "z", "velocity_x", "velocity_y", "velocity_z",
                "subjective_time",
                "excitation_count",
                "current_groundstate",
            ])

            for _, row in snapshot.iterrows():
                writer.writerow([
                    step,
                    row["position_x"],
                    row["position_y"],
                    row["position_z"],
                    row["velocity_x"],
                    row["velocity_y"],
                    row["velocity_z"],
                    float(row["subjective_time"]),
                    int(row["excitation_count"]),
                    int(row["current_groundstate"]),
                ])

        self.statusLabel.setText(
            f"Status: Wrote step {step} ({selected_time_s * 1000.0:.3f} ms) → {output_file}"
        )
        self.sampleCreated.emit(f"{output_file}@step_{step}")

def filter_step_snapshot(data_frame: pd.DataFrame, step_times_s: np.ndarray, step: int):
    """
    Helper for step-based selection.

    Parameters
    ----------
    data_frame:
        Source file data.
    step_times_s:
        Sorted unique subjective_time values in seconds.
    step:
        Integer step index.

    Returns
    -------
    pd.DataFrame | None
        Rows belonging to the selected step, or None if the step is invalid.
    """
    if data_frame.empty or step_times_s.size == 0:
        return None
    if step < 0 or step >= len(step_times_s):
        return None

    selected_time_s = float(step_times_s[step])
    filtered_data = data_frame[np.isclose(data_frame["subjective_time"], selected_time_s)]
    if filtered_data.empty:
        return None
    return filtered_data.sort_values("subjective_time").iloc[-1]