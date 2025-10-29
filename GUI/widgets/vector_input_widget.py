from PyQt5.QtWidgets  import QWidget, QHBoxLayout, QLineEdit
from PyQt5.QtGui      import QDoubleValidator
from PyQt5.QtCore     import pyqtSignal

class VectorInputWidget(QWidget):
    # Emit the new vector as a list of floats whenever it changes
    vectorChanged = pyqtSignal(list)

    def __init__(self, initial_value=None, parent=None):
        super().__init__(parent)
        if initial_value is None:
            initial_value = [0.0, 0.0, 0.0]

        self.layout = QHBoxLayout(self)
        self.edits = []
        validator = QDoubleValidator(-1e9, 1e9, 6, self)

        # Create three line edits for x, y, z
        for coord in initial_value:
            edit = QLineEdit(str(coord), self)
            edit.setValidator(validator)

            # hook each edit to the same slot:
            edit.textChanged.connect(self._on_any_text_changed)
            self.layout.addWidget(edit)
            self.edits.append(edit)

    def _on_any_text_changed(self, _):
        """
        Slot connected to each QLineEdit.textChanged. Attempts to
        read all three edits, convert them to floats, and emit the
        vectorChanged signal.
        """
        try:
            vec = [float(edit.text()) for edit in self.edits]
        except ValueError:
            # One of the edits isn’t a valid float yet → ignore
            return
        self.vectorChanged.emit(vec)

    def setVector(self, vec):
        """
        Programmatically update the three QLineEdits without
        emitting vectorChanged.
        """
        # block signals to avoid echoing back
        for edit in self.edits:
            edit.blockSignals(True)

        for edit, v in zip(self.edits, vec):
            edit.setText(str(v))

        for edit in self.edits:
            edit.blockSignals(False)