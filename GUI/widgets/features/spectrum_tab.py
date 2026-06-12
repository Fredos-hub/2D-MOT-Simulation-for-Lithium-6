from __future__ import annotations

import json

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QWidget, QFormLayout, QSpinBox, QDoubleSpinBox, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFileDialog, QLineEdit, QGroupBox, QComboBox, QCheckBox,
    QDialog,
)
from PyQt5.QtCore import Qt, QLocale, QDir

import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from GUI.widgets.common.vector_input_widget import VectorInputWidget
from src.spectrum_kernel import (
    BeamConfig,
    ATOM_NATURAL_LINEWIDTH,
    build_interaction,
    build_magnetic_field,
    compute_spectrum_scan,
)


# LaTeX-like math rendering for plot text (Computer Modern serif).
mpl.rcParams.setdefault("mathtext.fontset", "cm")
mpl.rcParams.setdefault("mathtext.rm", "serif")


# Per-axis fontsizes for the velocity-distribution panel.
VEL_TITLE_SIZE = 14
VEL_LABEL_SIZE = 13
VEL_TICK_SIZE = 11
VEL_SUPTITLE_SIZE = 15

# Per-axis fontsizes for the spectrum popup (a bit bigger — user request).
SPEC_TITLE_SIZE = 16
SPEC_LABEL_SIZE = 15
SPEC_TICK_SIZE = 13
SPEC_LEGEND_SIZE = 12

PLOT_FONT_FAMILY = "serif"

# Default simulation timestep used for the elapsed-time label.
DEFAULT_DT_S = 10e-6


# F, mF labels for Li6 ground states. 18-level models expose 6 ground states
# mapped to |F, mF⟩ via SimpleEighteenLevelInteraction (ground_mf + ground_gf).
LI6_GROUND_LABELS_18 = {
    0: r"$|F{=}1/2,\, m_F{=}{+}1/2\rangle$",
    1: r"$|F{=}3/2,\, m_F{=}{+}1/2\rangle$",
    2: r"$|F{=}1/2,\, m_F{=}{-}1/2\rangle$",
    3: r"$|F{=}3/2,\, m_F{=}{-}1/2\rangle$",
    4: r"$|F{=}3/2,\, m_F{=}{-}3/2\rangle$",
    5: r"$|F{=}3/2,\, m_F{=}{+}3/2\rangle$",
}
LI6_GROUND_LABELS_2 = {
    0: r"$|m_J{=}{-}1/2\rangle$",
    1: r"$|m_J{=}{+}1/2\rangle$",
}


def gs_label(gs_idx: int, n_ground_states: int) -> str:
    if n_ground_states == 6:
        return LI6_GROUND_LABELS_18.get(int(gs_idx), f"GS {int(gs_idx)}")
    if n_ground_states == 2:
        return LI6_GROUND_LABELS_2.get(int(gs_idx), f"GS {int(gs_idx)}")
    return f"GS {int(gs_idx)}"


def _style_axis(ax, *, title_size=VEL_TITLE_SIZE, label_size=VEL_LABEL_SIZE,
                tick_size=VEL_TICK_SIZE):
    """Apply LaTeX-like font styling and font sizes to an axis."""
    ax.title.set_fontsize(title_size)
    ax.title.set_fontfamily(PLOT_FONT_FAMILY)
    ax.xaxis.label.set_fontsize(label_size)
    ax.xaxis.label.set_fontfamily(PLOT_FONT_FAMILY)
    ax.yaxis.label.set_fontsize(label_size)
    ax.yaxis.label.set_fontfamily(PLOT_FONT_FAMILY)
    ax.tick_params(axis="both", which="major", labelsize=tick_size)
    for tlabel in ax.get_xticklabels() + ax.get_yticklabels():
        tlabel.set_fontfamily(PLOT_FONT_FAMILY)


INTERACTION_OPTIONS = [
    "Lithium6LevelInteraction",
    "Lithium18LevelInteraction",
    "SimpleEighteenLevelInteraction",
    "Lithium4LevelInteraction",
]


NORMALIZATION_OPTIONS = [
    "Total rate (Hz)",
    "Per atom (Hz)",
    "Per atom / Γ",
    "Normalized (peak = 1)",
]


class SpectrumTab(QWidget):
    """
    Tab for inspecting the velocity distribution of a simulation snapshot
    and configuring a spectroscopy probe beam for an absorption scan.

    The actual frequency-scan computation is wired through a placeholder
    `_compute_spectrum` so the physics path can be plugged in separately
    once the inputs are validated.
    """

    REQUIRED_FILE_COLUMNS = {
        "atom_id",
        "subjective_time",
        "position_x", "position_y", "position_z",
        "velocity_x", "velocity_y", "velocity_z",
        "excitation_count",
        "current_groundstate",
    }

    HELICITY_OPTIONS = ["-1", "0", "+1"]

    def __init__(self, parent=None):
        super().__init__(parent)

        self.file_df: pd.DataFrame | None = None
        self.step_times_s: np.ndarray = np.array([], dtype=float)
        self.b_field_config: dict | None = None

        self._init_ui()
        self._refresh_ui_state()

    # ------------------------------------------------------------------ UI

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        upper_layout = QHBoxLayout()
        upper_layout.addWidget(self._build_input_panel(), 1)
        upper_layout.addWidget(self._build_plot_panel(), 2)

        self.statusLabel = QLabel("Status: load a simulation result to begin.")

        main_layout.addLayout(upper_layout)
        main_layout.addWidget(self.statusLabel)

    def _build_input_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        layout.addWidget(self._build_snapshot_group())
        layout.addWidget(self._build_beam_group())
        layout.addWidget(self._build_scan_group())
        layout.addWidget(self._build_b_field_group())

        self.compute_btn = QPushButton("Compute Spectrum")
        self.compute_btn.clicked.connect(self._compute_spectrum)
        layout.addWidget(self.compute_btn)

        layout.addStretch()
        return panel

    def _build_snapshot_group(self) -> QGroupBox:
        group = QGroupBox("Simulation Snapshot")
        form = QFormLayout(group)

        self.input_line = QLineEdit()
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._browse_input)
        form.addRow("Input CSV:", self._hbox(self.input_line, self.browse_btn))

        self.step_spin = QSpinBox()
        self.step_spin.setRange(0, 0)
        self.step_spin.valueChanged.connect(self._on_step_changed)
        self.step_time_label = QLabel("0.000 ms")
        self.step_time_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.addRow("Step:", self._hbox(self.step_spin, self.step_time_label))

        self.bin_spin = QSpinBox()
        self.bin_spin.setRange(2, 1000)
        self.bin_spin.setValue(50)
        self.bin_spin.valueChanged.connect(lambda *_: self._update_velocity_plots())
        form.addRow("Bin count:", self.bin_spin)

        self.exclude_states_edit = QLineEdit()
        self.exclude_states_edit.setPlaceholderText("e.g. 0,2")
        self.exclude_states_edit.textChanged.connect(lambda *_: self._update_velocity_plots())
        form.addRow("Exclude states:", self.exclude_states_edit)

        return group

    def _build_beam_group(self) -> QGroupBox:
        group = QGroupBox("Spectroscopy Beam")
        form = QFormLayout(group)

        self.use_position_check = QCheckBox("Apply Gaussian beam profile at atom positions")
        self.use_position_check.setChecked(False)
        self.use_position_check.toggled.connect(self._on_use_position_toggled)
        form.addRow(self.use_position_check)

        self.beam_origin = VectorInputWidget([0.0, 0.0, 0.0])
        self.beam_origin.setEnabled(False)
        self.beam_direction = VectorInputWidget([0.0, 1.0, 0.0])
        form.addRow("Origin (m):", self.beam_origin)
        form.addRow("Direction:", self.beam_direction)

        self.beam_power_spin = self._make_double_spin(
            suffix=" mW", decimals=3, vmin=0.0, vmax=1e6, value=1.0,
        )
        form.addRow("Power:", self.beam_power_spin)

        self.beam_freq_spin = self._make_double_spin(
            suffix=" MHz", decimals=3, vmin=0.0, vmax=1e12, value=446799571.079,
        )
        form.addRow("Frequency:", self.beam_freq_spin)

        self.beam_detuning_spin = self._make_double_spin(
            suffix=" Γ", decimals=2, vmin=-1e6, vmax=1e6, value=0.0,
        )
        form.addRow("Detuning:", self.beam_detuning_spin)

        self.beam_handedness = QComboBox()
        self.beam_handedness.addItems(self.HELICITY_OPTIONS)
        self.beam_handedness.setCurrentText("+1")
        form.addRow("Handedness:", self.beam_handedness)

        self.beam_radius_spin = self._make_double_spin(
            suffix=" mm", decimals=2, vmin=0.0, vmax=1e4, value=5.0,
        )
        form.addRow("Radius (waist):", self.beam_radius_spin)

        self.interaction_combo = QComboBox()
        self.interaction_combo.addItems(INTERACTION_OPTIONS)
        self.interaction_combo.setCurrentText("Lithium18LevelInteraction")
        form.addRow("Interaction model:", self.interaction_combo)

        return group

    def _build_scan_group(self) -> QGroupBox:
        group = QGroupBox("Scan Range")
        form = QFormLayout(group)

        self.scan_min_spin = self._make_double_spin(
            suffix=" MHz", decimals=2, vmin=-1e6, vmax=1e6, value=-20.0,
        )
        self.scan_max_spin = self._make_double_spin(
            suffix=" MHz", decimals=2, vmin=-1e6, vmax=1e6, value=20.0,
        )
        form.addRow("Min:", self.scan_min_spin)
        form.addRow("Max:", self.scan_max_spin)

        self.scan_bin_spin = QSpinBox()
        self.scan_bin_spin.setRange(2, 5000)
        self.scan_bin_spin.setValue(100)
        form.addRow("Bin count:", self.scan_bin_spin)

        self.normalization_combo = QComboBox()
        self.normalization_combo.addItems(NORMALIZATION_OPTIONS)
        self.normalization_combo.setCurrentText("Per atom (Hz)")
        form.addRow("Normalization:", self.normalization_combo)

        self.show_per_gs_check = QCheckBox("Show per-groundstate components")
        self.show_per_gs_check.setChecked(True)
        form.addRow(self.show_per_gs_check)

        return group

    def _build_b_field_group(self) -> QGroupBox:
        group = QGroupBox("Magnetic Field")
        form = QFormLayout(group)

        self.b_field_check = QCheckBox("Consider magnetic field")
        self.b_field_check.toggled.connect(self._on_b_field_toggled)
        form.addRow(self.b_field_check)

        self.b_field_line = QLineEdit()
        self.b_field_line.setEnabled(False)
        self.b_field_browse_btn = QPushButton("Browse…")
        self.b_field_browse_btn.setEnabled(False)
        self.b_field_browse_btn.clicked.connect(self._browse_b_field)
        form.addRow("parameters.json:", self._hbox(self.b_field_line, self.b_field_browse_btn))

        self.b_field_info_label = QLabel("No file loaded.")
        self.b_field_info_label.setWordWrap(True)
        form.addRow("Info:", self.b_field_info_label)

        return group

    def _build_plot_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(7, 6), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.addWidget(self.toolbar, 1)
        self.export_snapshot_btn = QPushButton("Export data to CSV")
        self.export_snapshot_btn.setEnabled(False)
        self.export_snapshot_btn.clicked.connect(self._export_snapshot_csv)
        toolbar_row.addWidget(self.export_snapshot_btn, 0)

        layout.addLayout(toolbar_row)
        layout.addWidget(self.canvas)

        self.axes = self.fig.subplots(2, 2)
        self._reset_axis_titles()
        for ax in self.axes.flat:
            ax.grid(True, alpha=0.25)
            _style_axis(ax)

        self.canvas.draw_idle()
        return widget

    def _make_double_spin(self, suffix, decimals, vmin, vmax, value) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(vmin, vmax)
        spin.setDecimals(decimals)
        spin.setSuffix(suffix)
        spin.setLocale(QLocale(QLocale.C))
        spin.setValue(value)
        return spin

    def _hbox(self, *widgets):
        h = QHBoxLayout()
        for w in widgets:
            h.addWidget(w)
        return h

    # ------------------------------------------------------------ snapshot

    def _refresh_ui_state(self):
        has_file = self.file_df is not None and self.step_times_s.size > 0
        self.step_spin.setEnabled(has_file)
        self.bin_spin.setEnabled(has_file)
        self.compute_btn.setEnabled(has_file)

        if has_file:
            max_step = max(0, len(self.step_times_s) - 1)
            self.step_spin.setMaximum(max_step)
            self.step_spin.setValue(min(self.step_spin.value(), max_step))
            time_ms = self._elapsed_time_s(self.step_spin.value()) * 1000.0
            self.step_time_label.setText(f"{time_ms:.3f} ms")
            self._update_velocity_plots()
        else:
            self.step_spin.setMaximum(0)
            self.step_spin.setValue(0)
            self.step_time_label.setText("0.000 ms")
            self._clear_plots()

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Simulation Result", filter="CSV files (*.csv)"
        )
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
            self.statusLabel.setText(f"Status: failed to read input file ({exc})")
            self._refresh_ui_state()
            return

        missing = self.REQUIRED_FILE_COLUMNS - set(df.columns)
        if missing:
            self.file_df = None
            self.step_times_s = np.array([], dtype=float)
            self.statusLabel.setText(
                f"Status: missing required columns: {', '.join(sorted(missing))}"
            )
            self._refresh_ui_state()
            return

        self.file_df = df.copy()
        self.step_times_s = np.array(
            sorted(self.file_df["subjective_time"].dropna().unique()), dtype=float
        )

        if self.step_times_s.size == 0:
            self.statusLabel.setText("Status: input file has no valid subjective_time values.")
        else:
            self.statusLabel.setText(
                f"Status: loaded {len(df)} rows, {len(self.step_times_s)} steps."
            )
        self._refresh_ui_state()

    def _on_step_changed(self, step: int):
        if self.step_times_s.size == 0:
            self.step_time_label.setText("0.000 ms")
            return
        step = max(0, min(step, len(self.step_times_s) - 1))
        time_ms = self._elapsed_time_s(step) * 1000.0
        self.step_time_label.setText(f"{time_ms:.3f} ms")
        self._update_velocity_plots()

    def _parse_excluded_states(self) -> set[int]:
        # Lenient parse: ignore non-integer tokens silently so plots still
        # update while the user is mid-typing.
        excluded: set[int] = set()
        for token in self.exclude_states_edit.text().split(","):
            token = token.strip()
            if not token:
                continue
            try:
                excluded.add(int(token))
            except ValueError:
                continue
        return excluded

    def _step_snapshot(self) -> pd.DataFrame | None:
        if self.file_df is None or self.step_times_s.size == 0:
            return None
        step = int(self.step_spin.value())
        if not (0 <= step < len(self.step_times_s)):
            return None
        selected_time_s = float(self.step_times_s[step])
        slice_df = self.file_df[
            np.isclose(self.file_df["subjective_time"], selected_time_s)
        ]
        if slice_df.empty:
            return None
        # Keep the last row per atom in case of duplicates at the same step.
        snapshot = (
            slice_df.sort_values(["atom_id", "subjective_time"])
            .groupby("atom_id", as_index=False)
            .tail(1)
        )
        excluded = self._parse_excluded_states()
        if excluded:
            snapshot = snapshot[~snapshot["current_groundstate"].isin(excluded)]
            if snapshot.empty:
                return None
        return snapshot

    # ---------------------------------------------------------------- plot

    def _reset_axis_titles(self):
        self.axes[0, 0].set_title(r"$v_x$")
        self.axes[0, 1].set_title(r"$v_y$")
        self.axes[1, 0].set_title(r"$v_z$")
        self.axes[1, 1].set_title(r"$|v|$")

    def _clear_plots(self):
        for ax in self.axes.flat:
            ax.clear()
            ax.grid(True, alpha=0.25)
        self._reset_axis_titles()
        for ax in self.axes.flat:
            _style_axis(ax)
        self.fig.suptitle("")
        self.canvas.draw_idle()

    def _elapsed_time_s(self, step: int) -> float:
        """Lab-frame elapsed time for a snapshot step: (step + 1) * dt."""
        return (int(step) + 1) * DEFAULT_DT_S

    def _update_velocity_plots(self):
        snapshot = self._step_snapshot()
        if snapshot is None:
            self.export_snapshot_btn.setEnabled(False)
            self._clear_plots()
            return

        bins = int(self.bin_spin.value())
        vx = snapshot["velocity_x"].to_numpy()
        vy = snapshot["velocity_y"].to_numpy()
        vz = snapshot["velocity_z"].to_numpy()
        v_abs = np.sqrt(vx**2 + vy**2 + vz**2)

        time_ms = self._elapsed_time_s(self.step_spin.value()) * 1000.0

        for ax in self.axes.flat:
            ax.clear()
            ax.grid(True, alpha=0.25)

        self._plot_hist(self.axes[0, 0], vx, bins, r"$v_x$")
        self._plot_hist(self.axes[0, 1], vy, bins, r"$v_y$")
        self._plot_hist(self.axes[1, 0], vz, bins, r"$v_z$")
        self._plot_hist(self.axes[1, 1], v_abs, bins, r"$|v|$")

        excluded = self._parse_excluded_states()
        suffix = (
            f", excluded states: {{{', '.join(str(s) for s in sorted(excluded))}}}"
            if excluded
            else ""
        )
        self.fig.suptitle(
            rf"Velocity distribution at $t = {time_ms:.3f}$ ms ($N = {len(snapshot)}$){suffix}",
            fontsize=VEL_SUPTITLE_SIZE,
            fontfamily=PLOT_FONT_FAMILY,
        )
        self.export_snapshot_btn.setEnabled(True)
        self.canvas.draw_idle()

    def _plot_hist(self, ax, data: np.ndarray, bins: int, label: str):
        if data.size == 0:
            ax.set_title(f"{label} (empty)")
            _style_axis(ax)
            return
        ax.hist(data, bins=bins, alpha=0.85, edgecolor="black", linewidth=0.3)
        ax.set_title(label)
        ax.set_xlabel(r"m/s")
        ax.set_ylabel(r"Count")
        _style_axis(ax)

    # ------------------------------------------------------ snapshot export

    def _export_snapshot_csv(self):
        snapshot = self._step_snapshot()
        if snapshot is None:
            self.statusLabel.setText("Status: no snapshot to export.")
            return

        step = int(self.step_spin.value())
        time_ms = self._elapsed_time_s(step) * 1000.0
        path, _ = QFileDialog.getSaveFileName(
            self, "Export snapshot velocity data",
            f"velocity_snapshot_step{step}.csv",
            filter="CSV files (*.csv)",
        )
        if not path:
            return

        vx = snapshot["velocity_x"].to_numpy()
        vy = snapshot["velocity_y"].to_numpy()
        vz = snapshot["velocity_z"].to_numpy()
        v_abs = np.sqrt(vx**2 + vy**2 + vz**2)

        out = pd.DataFrame({
            "atom_id": snapshot["atom_id"].to_numpy(),
            "current_groundstate": snapshot["current_groundstate"].to_numpy(),
            "velocity_x": vx,
            "velocity_y": vy,
            "velocity_z": vz,
            "velocity_abs": v_abs,
        })
        out.attrs["step"] = step
        out.attrs["elapsed_time_ms"] = time_ms
        try:
            out.to_csv(path, index=False)
        except Exception as exc:
            self.statusLabel.setText(f"Status: failed to write CSV ({exc}).")
            return
        self.statusLabel.setText(f"Status: snapshot exported ({len(out)} rows) → {path}")

    # ----------------------------------------------------- magnetic field

    def _on_use_position_toggled(self, checked: bool):
        self.beam_origin.setEnabled(checked)

    def _on_b_field_toggled(self, checked: bool):
        self.b_field_line.setEnabled(checked)
        self.b_field_browse_btn.setEnabled(checked)
        if not checked:
            self.b_field_config = None
            self.b_field_info_label.setText("No file loaded.")

    def _browse_b_field(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open parameters.json", filter="JSON files (*.json)"
        )
        if not path:
            return
        rel = QDir().relativeFilePath(path)
        self.b_field_line.setText(rel)
        self._load_b_field_config(path)

    def _load_b_field_config(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            self.b_field_config = None
            self.b_field_info_label.setText(f"Failed to read: {exc}")
            return

        b_block = data.get("Magnetic_Fields")
        if not isinstance(b_block, dict):
            self.b_field_config = None
            self.b_field_info_label.setText("No 'Magnetic_Fields' block in file.")
            return

        self.b_field_config = b_block
        b_type = b_block.get("type", "unknown")
        self.b_field_info_label.setText(f"Loaded B-field type: {b_type}")

    # --------------------------------------------------------- compute stub

    def _read_vector(self, widget: VectorInputWidget) -> list[float]:
        out: list[float] = []
        for edit in widget.edits:
            text = edit.text().strip()
            try:
                out.append(float(text))
            except ValueError:
                out.append(0.0)
        return out

    def _compute_spectrum(self):
        """
        Run the steady-state frequency-scan and show the result in a popup.
        """
        snapshot = self._step_snapshot()
        if snapshot is None:
            self.statusLabel.setText("Status: no snapshot available for the selected step.")
            return

        scan_min = self.scan_min_spin.value()
        scan_max = self.scan_max_spin.value()
        if scan_min >= scan_max:
            self.statusLabel.setText("Status: scan min must be strictly less than scan max.")
            return

        n_bins = int(self.scan_bin_spin.value())

        use_position = self.use_position_check.isChecked()
        beam = BeamConfig(
            origin_m=np.array(self._read_vector(self.beam_origin), dtype=np.float64),
            direction=np.array(self._read_vector(self.beam_direction), dtype=np.float64),
            power_W=self.beam_power_spin.value() * 1e-3,
            frequency_Hz=self.beam_freq_spin.value() * 1e6,
            detuning_offset_rad=self.beam_detuning_spin.value() * ATOM_NATURAL_LINEWIDTH,
            handedness=int(self.beam_handedness.currentText()),
            radius_m=self.beam_radius_spin.value() * 1e-3,
            use_position=use_position,
        )

        use_b = self.b_field_check.isChecked()
        if use_b and self.b_field_config is None:
            self.statusLabel.setText(
                "Status: magnetic field is enabled but no parameters.json is loaded."
            )
            return

        try:
            interaction = build_interaction(self.interaction_combo.currentText())
        except Exception as exc:
            self.statusLabel.setText(f"Status: failed to build interaction ({exc}).")
            return

        try:
            mag_field = build_magnetic_field(self.b_field_config if use_b else None)
        except Exception as exc:
            self.statusLabel.setText(f"Status: failed to build magnetic field ({exc}).")
            return

        positions = snapshot[["position_x", "position_y", "position_z"]].to_numpy(dtype=np.float64)
        velocities = snapshot[["velocity_x", "velocity_y", "velocity_z"]].to_numpy(dtype=np.float64)
        ground_states = snapshot["current_groundstate"].to_numpy(dtype=np.int32)

        detunings_MHz = np.linspace(scan_min, scan_max, n_bins)
        self.statusLabel.setText("Status: computing spectrum…")
        self.compute_btn.setEnabled(False)
        try:
            result = compute_spectrum_scan(
                positions=positions,
                velocities=velocities,
                ground_states=ground_states,
                interaction=interaction,
                magnetic_field=mag_field,
                beam=beam,
                detunings_MHz=detunings_MHz,
            )
        except Exception as exc:
            self.statusLabel.setText(f"Status: spectrum computation failed ({exc}).")
            self.compute_btn.setEnabled(True)
            return
        finally:
            self.compute_btn.setEnabled(True)

        scale, y_label = self._scaling_for_total(result.rates, result.n_atoms)
        y_total = result.rates * scale
        y_per_gs = result.rates_per_groundstate * scale  # same scaling so curves stack to total

        time_ms = self._elapsed_time_s(self.step_spin.value()) * 1000.0
        excluded = self._parse_excluded_states()
        title = (
            rf"Spectrum at $t = {time_ms:.3f}$ ms ($N = {result.n_atoms}$)"
            f"{', exclude ' + ','.join(str(s) for s in sorted(excluded)) if excluded else ''}"
        )

        show_per_gs = self.show_per_gs_check.isChecked()
        dlg = SpectrumDialog(
            self,
            x=result.detunings_MHz,
            y_total=y_total,
            y_per_gs=y_per_gs if show_per_gs else None,
            groundstates=result.groundstates,
            counts_per_gs=result.counts_per_groundstate,
            n_ground_states=int(interaction.number_of_ground_states),
            y_label=y_label,
            title=title,
        )
        dlg.show()
        self._last_spectrum_dialog = dlg  # keep reference so the non-modal dialog stays alive

        self.statusLabel.setText(
            f"Status: scan done — {result.n_atoms} atoms × {n_bins} points "
            f"({scan_min:.2f}…{scan_max:.2f} MHz)."
        )

    def _scaling_for_total(self, rates: np.ndarray, n_atoms: int) -> tuple[float, str]:
        """
        Choose a single scalar factor to scale the total rate (and apply the
        same factor to per-groundstate components so they stack to the total).
        """
        mode = self.normalization_combo.currentText()
        if mode == "Total rate (Hz)":
            return 1.0, "Total scattering rate (photons/s)"
        if mode == "Per atom (Hz)":
            return 1.0 / max(n_atoms, 1), "Mean scattering rate per atom (photons/s)"
        if mode == "Per atom / Γ":
            return 1.0 / (max(n_atoms, 1) * ATOM_NATURAL_LINEWIDTH), "Mean scattering rate per atom / Γ"
        if mode == "Normalized (peak = 1)":
            peak = float(np.max(rates))
            if peak <= 0.0:
                return 1.0, "Normalized (peak = 1) — peak is zero"
            return 1.0 / peak, "Normalized (peak = 1)"
        return 1.0, "Rate"


class SpectrumDialog(QDialog):
    """Non-modal popup showing a computed spectrum scan."""

    def __init__(
        self,
        parent,
        x: np.ndarray,
        y_total: np.ndarray,
        y_per_gs: np.ndarray | None,
        groundstates: np.ndarray,
        counts_per_gs: np.ndarray,
        n_ground_states: int,
        y_label: str,
        title: str,
    ):
        super().__init__(parent)
        self.setWindowTitle("Spectrum")
        self.setModal(False)
        self.resize(900, 540)

        # Stash data for CSV export.
        self._x = np.asarray(x)
        self._y_total = np.asarray(y_total)
        self._y_per_gs = None if y_per_gs is None else np.asarray(y_per_gs)
        self._groundstates = np.asarray(groundstates, dtype=int)
        self._counts_per_gs = np.asarray(counts_per_gs, dtype=int)
        self._n_ground_states = int(n_ground_states)
        self._y_label = y_label

        layout = QVBoxLayout(self)

        fig = Figure(figsize=(8, 4.5), tight_layout=True)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)

        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.addWidget(toolbar, 1)
        self.export_btn = QPushButton("Export data to CSV")
        self.export_btn.clicked.connect(self._export_csv)
        toolbar_row.addWidget(self.export_btn, 0)

        layout.addLayout(toolbar_row)
        layout.addWidget(canvas)

        ax = fig.add_subplot(111)

        # Per-groundstate components first (thin, beneath the total).
        if self._y_per_gs is not None and self._y_per_gs.shape[0] > 1:
            for k, gs in enumerate(self._groundstates):
                lbl = gs_label(int(gs), self._n_ground_states)
                ax.plot(
                    x, self._y_per_gs[k],
                    linewidth=1.2, alpha=0.85,
                    label=rf"{lbl}  ($N={int(self._counts_per_gs[k])}$)",
                )

        ax.plot(x, y_total, linewidth=2.2, color="black", label="Total")
        ax.fill_between(x, y_total, alpha=0.12, color="black")

        ax.set_title(title)
        ax.set_xlabel(r"Probe detuning (MHz)")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=SPEC_LEGEND_SIZE, frameon=False,
                  prop={"family": PLOT_FONT_FAMILY})
        _style_axis(
            ax,
            title_size=SPEC_TITLE_SIZE,
            label_size=SPEC_LABEL_SIZE,
            tick_size=SPEC_TICK_SIZE,
        )

        canvas.draw_idle()

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export spectrum data", "spectrum.csv",
            filter="CSV files (*.csv)",
        )
        if not path:
            return

        cols = {"detuning_MHz": self._x, "total": self._y_total}
        if self._y_per_gs is not None:
            for k, gs in enumerate(self._groundstates):
                col_name = f"groundstate_{int(gs)}_N{int(self._counts_per_gs[k])}"
                cols[col_name] = self._y_per_gs[k]
        out = pd.DataFrame(cols)
        try:
            out.to_csv(path, index=False)
        except Exception as exc:
            self.export_btn.setText(f"Export failed: {exc}")
            return
        self.export_btn.setText("Export data to CSV (saved)")
