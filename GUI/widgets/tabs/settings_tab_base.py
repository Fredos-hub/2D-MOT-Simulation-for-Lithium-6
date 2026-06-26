from collections.abc import Iterator
from contextlib import contextmanager

from PyQt5.QtWidgets import QWidget


@contextmanager
def signals_blocked(*objects) -> Iterator[None]:
    """Block Qt signals on each object for the duration of the block, then restore."""
    for obj in objects:
        obj.blockSignals(True)
    try:
        yield
    finally:
        for obj in objects:
            obj.blockSignals(False)


class SettingsTab(QWidget):
    """Base for FileModel-backed settings tabs.

    Subclasses set ``SECTION`` (the JSON section they edit) and implement
    ``_init_ui`` and ``_connect_signals``. Editing widgets write back through
    ``_update_model``; ``setModel`` populates widgets inside a ``signals_blocked``
    block so loading never marks the model dirty.
    """

    SECTION = None

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._model = None
        self._init_ui()
        self._connect_signals()

    def _init_ui(self) -> None:
        raise NotImplementedError

    def _connect_signals(self) -> None:
        raise NotImplementedError

    def _update_model(self, key, value) -> None:
        if self._model is not None:
            self._model.set(value, self.SECTION, key)
