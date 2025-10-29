from PyQt5.QtWidgets import QToolBar, QAction, QStyle
from PyQt5.QtGui import QIcon

class ToolBar(QToolBar):
    def __init__(self, parent=None):
        super().__init__("Main Toolbar", parent)
        self.setMovable(False)
        self.create_actions()
        self.add_actions()

    def create_actions(self):
        # Toolbar action for loading a file
        self.load_action = QAction(self.style().standardIcon(QStyle.SP_DialogOpenButton), "Open Directory", self)
        self.load_action.setShortcut("Ctrl+L")
        self.load_action.setStatusTip("Open Directorx")

        # Toolbar action for creating a new file
        self.new_action = QAction(self.style().standardIcon(QStyle.SP_FileIcon), "New", self)
        self.new_action.setShortcut("Ctrl+N")
        self.new_action.setStatusTip("Create a new file")

        # Toolbar action for saving a file
        self.save_action = QAction(self.style().standardIcon(QStyle.SP_DialogSaveButton), "Save", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.setStatusTip("Save the current file")
        self.save_action.setEnabled(False)  # Initially disabled

        self.save_all_action = QAction(self.style().standardIcon(QStyle.SP_DialogSaveAllButton), "Save All", self)
        self.save_all_action.setShortcut("Ctrl+Shift+S")
        self.save_all_action.setStatusTip("Save all files")
        self.save_all_action.setEnabled(False)


        self.discard_action = QAction(self.style().standardIcon(QStyle.SP_DialogOkButton), "Discard Changes", self)
        self.discard_action.setShortcut("Ctrl + D")
        self.discard_action.setStatusTip("Discard Changes to the current file")
        self.discard_action.setEnabled(False)


        self.discard_all_action = QAction(self.style().standardIcon(QStyle.SP_DialogResetButton), "Discard All Changes", self)
        self.discard_all_action.setShortcut("Ctrl + Shift + D")
        self.discard_all_action.setStatusTip("Discard All Changes")
        self.discard_all_action.setEnabled(False)

        # Toolbar action for running simulation
        self.run_action = QAction(self.style().standardIcon(QStyle.SP_MediaPlay), "Run", self)
        self.run_action.setShortcut("Ctrl+R")
        self.run_action.setStatusTip("Run the current file")
        self.run_action.setEnabled(True)  # Initially disabled

        # Toolbar action for pausing simulation
        self.pause_action = QAction(self.style().standardIcon(QStyle.SP_MediaPause), "Pause", self)
        self.pause_action.setShortcut("Ctrl+P")
        self.pause_action.setStatusTip("Pause the current simulation")
        self.pause_action.setEnabled(False)  # Initially disabled

        # Toolbar action for resuming simulation
        self.resume_action = QAction(self.style().standardIcon(QStyle.SP_MediaSeekForward), "Resume", self)
        self.resume_action.setShortcut("Ctrl+Shift+R")
        self.resume_action.setStatusTip("Resume the current simulation")
        self.resume_action.setEnabled(False)  # Initially disabled


        # Toolbar action for canceling simulation
        self.cancel_action = QAction(self.style().standardIcon(QStyle.SP_DialogCancelButton), "Cancel", self)
        self.cancel_action.setShortcut("Ctrl+C")
        self.cancel_action.setStatusTip("Cancel the current operation")
        self.cancel_action.setEnabled(False)

    def add_actions(self):
        # Add actions to the toolbar in the desired order
        self.addAction(self.load_action)
        self.addAction(self.new_action)
        self.addSeparator()
        self.addAction(self.save_action)
        self.addAction(self.save_all_action)
        self.addAction(self.discard_action)
        self.addAction(self.discard_all_action)
        self.addSeparator()
        self.addAction(self.run_action)
        self.addAction(self.pause_action)
        self.addAction(self.resume_action)
        self.addAction(self.cancel_action)