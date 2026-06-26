from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlottingTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        label = QLabel("Plotting Tab - Shell", self)
        layout.addWidget(label)
