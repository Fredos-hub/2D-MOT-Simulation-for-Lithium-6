from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTableWidget,
    QHeaderView, QSizePolicy, QLabel, QDoubleSpinBox, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt, QLocale
from GUI.widgets.vector_input_widget import VectorInputWidget

class BarDipolesTable(QWidget):
    """
    Table to configure bar‑dipoles in Magnetic_Fields → dipoles.
    """
    def __init__(self, model=None, parent=None):
        super().__init__(parent)
        self._model = None
        self._init_ui()
        if model is not None:
            self.setModel(model)

    def _init_ui(self):
        layout = QHBoxLayout(self)

        # Left: add/remove & plot buttons, with separator
        panel = QWidget()
        pnl_layout = QVBoxLayout(panel)

        # Group 1: Add / Remove
        btn_add    = QPushButton("Add Dipole")
        btn_remove = QPushButton("Remove Selected")
        btn_add.clicked.connect(self._add_new_dipole)
        btn_remove.clicked.connect(self._remove_selected)
        pnl_layout.addWidget(btn_add)
        pnl_layout.addWidget(btn_remove)

        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        pnl_layout.addWidget(sep)

        # Group 2: Plot buttons
        btn_plot_ax = QPushButton("Plot Field Along Axis")
        btn_plot_pl = QPushButton("Plot Field in Plane")
        btn_plot_ax.clicked.connect(self._plot_along_axis)
        btn_plot_pl.clicked.connect(self._plot_field_in_plane)
        pnl_layout.addWidget(btn_plot_ax)
        pnl_layout.addWidget(btn_plot_pl)

        pnl_layout.addStretch()
        panel.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        # Right: the dipoles table
        self.table = QTableWidget(0, 4)
        self.table.setMinimumWidth(700)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        hdr = self.table.horizontalHeader()
        # Stretch all columns to fill
        for i in range(4):
            hdr.setSectionResizeMode(i, QHeaderView.Stretch)

        self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.setHorizontalHeaderLabels([
            "Position (m)",
            "Dimension (m)",
            "Orientation",
            "Magnetization (×10⁵ A/m)"
        ])

        layout.addWidget(panel)
        layout.addWidget(self.table)

    def setModel(self, model):
        """Load dipoles list from model."""
        self._model = model
        dip_list = model.get('Magnetic_Fields', 'dipoles', default=[]) or []
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(dip_list))
            for row, cfg in enumerate(dip_list):
                self._populate_row(row, cfg)
            self.table.setVerticalHeaderLabels([f"D{n}" for n in range(len(dip_list))])
        finally:
            self.table.blockSignals(False)

    def _populate_row(self, row, cfg):
        # Position widget
        pos_w = VectorInputWidget(cfg.get('position', [0,0,0]), self)
        pos_w.vectorChanged.connect(lambda vec, r=row: self._update_field(r, 'position', vec))
        self.table.setCellWidget(row, 0, pos_w)

        # Dimension widget
        dim_w = VectorInputWidget(cfg.get('dimension', [0.01,0.01,0.01]), self)
        dim_w.vectorChanged.connect(lambda vec, r=row: self._update_field(r, 'dimension', vec))
        self.table.setCellWidget(row, 1, dim_w)

        # Orientation widget
        ori_w = VectorInputWidget(cfg.get('orientation', [1,0,0]), self)
        ori_w.vectorChanged.connect(lambda vec, r=row: self._update_field(r, 'orientation', vec))
        self.table.setCellWidget(row, 2, ori_w)

        # Magnetization spinbox (display in 1e5 A/m, decimal=2, dot separator)
        raw_M = cfg.get('magnetization', 8.8e5)
        disp_M = raw_M / 1e5

        mag_spin = QDoubleSpinBox()
        mag_spin.setLocale(QLocale(QLocale.C))
        mag_spin.setRange(0.0, 100.0)
        mag_spin.setDecimals(2)
        mag_spin.setSingleStep(0.10)
        mag_spin.setSuffix("·10⁵")
        mag_spin.setValue(disp_M)
        mag_spin.valueChanged.connect(
            lambda v, r=row: self._update_field(r, 'magnetization', v * 1e5)
        )
        self.table.setCellWidget(row, 3, mag_spin)

    def _add_new_dipole(self):
        if not self._model:
            return
        lst = list(self._model.get('Magnetic_Fields','dipoles', default=[]) or [])
        lst.append({
            'position':     [0.0, 0.0, 0.0],
            'dimension':    [0.01,0.01,0.01],
            'orientation':  [1.0, 0.0, 0.0],
            'magnetization': 8.8e5
        })
        self._model.set(lst, 'Magnetic_Fields', 'dipoles')
        self.setModel(self._model)

    def _remove_selected(self):
        """Remove the currently highlighted dipole row."""
        if not self._model:
            return
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.warning(self, "Remove Dipole", "No row selected to remove.")
            return
        lst = list(self._model.get('Magnetic_Fields','dipoles', default=[]) or [])
        if 0 <= row < len(lst):
            lst.pop(row)
            self._model.set(lst, 'Magnetic_Fields', 'dipoles')
            self.setModel(self._model)

    def _update_field(self, row, key, value):
        """Update one field in the dipole list and write back."""
        if not self._model:
            return
        lst = list(self._model.get('Magnetic_Fields','dipoles', default=[]) or [])
        if not (0 <= row < len(lst)):
            return
        entry = dict(lst[row])
        entry[key] = value
        lst[row] = entry
        self._model.set(lst, 'Magnetic_Fields', 'dipoles')

    def _plot_along_axis(self):
        QMessageBox.information(self, "Plot Along Axis", "Plotting along axis…")

    def _plot_field_in_plane(self):
        QMessageBox.information(self, "Plot Field in Plane", "Plotting field in plane…")