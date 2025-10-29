from PyQt5.QtWidgets import (
    QWidget, QFormLayout, QSpinBox, QDoubleSpinBox, QHBoxLayout, QVBoxLayout,
    QRadioButton, QProgressBar, QStackedWidget, QLabel, QPushButton,
    QListWidget, QFileDialog, QLineEdit, QGroupBox, QComboBox
)
from PyQt5.QtCore import pyqtSignal, Qt, QLocale,QDir
from GUI.oven_worker import OvenWorker
import pandas as pd
import csv
class SampleGeneratorTab(QWidget):
    # Signals emitted when a sample is generated
    sampleCreated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
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

        # Right side: placeholder for graphs
        self.graph_area = QLabel("Graph Area (to be implemented)")
        self.graph_area.setAlignment(Qt.AlignCenter)
        upper_layout.addWidget(self.graph_area, 2)

        # Progress and status
        self.dist_progress_bar = QProgressBar(value=0)
        lower_layout.addWidget(self.dist_progress_bar)
        self.statusLabel = QLabel("Status: Not started")
        lower_layout.addWidget(self.statusLabel)

        main_layout.addLayout(upper_layout)
        main_layout.addLayout(lower_layout)

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

        # Oven geometry: radius and y-position (1 decimal)
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

        # Aperture geometry: list + input fields (1 decimal)
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

        # Output file directory
        self.output_line = QLineEdit()
        self.browse_output_btn = QPushButton("Browse…")
        self.browse_output_btn.clicked.connect(self._browse_output)
        layout.addRow("Output file:", self._hbox(self.output_line, self.browse_output_btn))

        # Generate button
        self.generate_oven_btn = QPushButton("Generate Sample")
        self.generate_oven_btn.clicked.connect(self._generate_from_oven)
        layout.addRow("", self.generate_oven_btn)

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

        # Time selector
        self.time_spin = QDoubleSpinBox()
        self.time_spin.setRange(0, 1e6)
        self.time_spin.setSuffix(" ms")
        self.time_spin.setValue(1.5)
        self.time_spin.setLocale(QLocale(QLocale.C))
        layout.addRow("Time (subjective):", self.time_spin)

        # Output file directory
        self.output_line_2 = QLineEdit()
        self.browse_output_btn_2 = QPushButton("Browse…")
        self.browse_output_btn_2.clicked.connect(self._browse_output)
        layout.addRow("Output file:", self._hbox(self.output_line_2, self.browse_output_btn_2))

        # Generate button
        self.generate_file_btn = QPushButton("Generate Sample")
        self.generate_file_btn.clicked.connect(self._generate_from_file)
        layout.addRow("", self.generate_file_btn)

        group.setLayout(layout)
        return group

    def _hbox(self, *widgets):
        h = QHBoxLayout()
        for w in widgets:
            h.addWidget(w)
        return h

    def _on_mode_changed(self, checked):
        idx = 0 if self.oven_radio.isChecked() else 1
        self.stacked.setCurrentIndex(idx)

    def _add_aperture(self):
        r = self.ap_radius_spin.value()
        y = self.ap_ypos_spin.value()
        self.aperture_data.append((r,y))
        self.aperture_list.addItem(f"Aperture: r={r:.1f} mm, y={y:.1f} mm")

    def _remove_aperture(self):
        for item in self.aperture_list.selectedItems():
            row = self.aperture_list.row(item)
            self.aperture_data.pop(row)
            self.aperture_list.takeItem(row)

    def _browse_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Output", filter="CSV files (*.csv)")
        if path:
            rel = QDir().relativeFilePath(path)  # relative to the app’s cwd
            if self.oven_radio.isChecked():
                self.output_line.setText(rel)
            else:
                self.output_line_2.setText(rel)

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Sample File", filter="CSV files (*.csv)")
        if path:
            rel = QDir().relativeFilePath(path)
            self.input_line.setText(rel)


    def _generate_from_oven(self):
        # 1) read all the UI state into a plain dict
        params = {
            'atom_mass':         self.atom_mass_spin.value(),
            'distribution':      self.distribution_combo.currentText(),
            'num_atoms':         self.number_of_atoms_spin.value(),
            'temperature':       self.temp_spin.value(),
            'vmin':              self.vel_min.value(),
            'vmax':              self.vel_max.value(),
            'oven_geometry':     (self.oven_radius_spin.value(),
                                  self.oven_ypos_spin.value()),
            'apertures':         self.aperture_data, 
            'output_file':       self.output_line.text()
                    }

        # 2) disable button so user can’t double‐click
        self.generate_oven_btn.setEnabled(False)

        # 3) spin up worker thread
        self.worker = OvenWorker(params)
        self.worker.progress.connect(self.dist_progress_bar.setValue)
        self.worker.finished.connect(self._on_oven_finished)
        self.worker.start()

        self.statusLabel.setText("Status: Running…")

    def _on_oven_finished(self, filename):
        self.generate_oven_btn.setEnabled(True)
        self.statusLabel.setText(f"Status: Done → {filename}")
        self.sampleCreated.emit(filename)

    def _generate_from_file(self):
        input_file   = self.input_line.text()
        output_file  = self.output_line_2.text()
        t_threshold  = self.time_spin.value()*1e-3

        # read once
        df = pd.read_csv(input_file)

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'x','y','z','vx','vy','vz',
                'subjective_time','excitation_count','current_groundstate'
            ])

            # group & process each atom_id
            for atom_id, group in df.groupby("atom_id"):
                # keep only times ≤ threshold
                g = group[group["subjective_time"] <= t_threshold]
                if g.empty:
                    continue

                # pick the last (max-time) row
                row = g.loc[g["subjective_time"].idxmax()]

                # compute time‐to‐threshold
                dt = t_threshold - float(row["subjective_time"])

                # positions at t_threshold
                x  = float(row["position_x"] + row["velocity_x"] * dt)
                y  = float(row["position_y"] + row["velocity_y"] * dt)
                z  = float(row["position_z"] + row["velocity_z"] * dt)

                # velocity components (at the last record)
                vx = float(row["velocity_x"])
                vy = float(row["velocity_y"])
                vz = float(row["velocity_z"])

                # other metadata
                t  = float(t_threshold)
                exc = int(row["excitation_count"])
                gs  = int(row["current_groundstate"])

                writer.writerow([x, y, z, vx, vy, vz, t, exc, gs])

        print("worked so far")
        self.sampleCreated.emit(f"{input_file}@{t_threshold}ms")

def filter_time_threshold(data_frame: pd.DataFrame, threshold):

    filtered_data = data_frame[data_frame["subjective_time"] <= threshold]
    if not filtered_data.empty:
        return filtered_data.sort_values("subjective_time").iloc[-1]
        
