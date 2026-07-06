"""Offline diagonalizer-table generator worker (phase 04-05).

Clone of ``OvenWorker``: a ``QThread`` that drives
``diagonalizer_setup.generate_table`` off the GUI thread and reports via
``progress``/``finished``/``error``/``cancelled``, deleting a partial NPZ on
cancel.

``generate_table`` currently reproduces the fixed Li-6 D2 line from the frozen
``@njit`` helpers, so the per-manifold ``(I, J, g_J, A_hfs, B_hfs)`` params are
accepted for the future general D-05 tool but validated against Li-6 D2 for
now (a mismatch raises rather than silently mislabelling the output).
"""
import os

from PyQt5.QtCore import QThread, pyqtSignal

import src.diagonalizer_setup as ds


class DiagonalizerWorker(QThread):
    progress = pyqtSignal(int)  # 0–100
    finished = pyqtSignal(str)  # output NPZ path
    error = pyqtSignal(str)  # error message; finished is NOT emitted
    cancelled = pyqtSignal()  # emits when the user aborts

    def __init__(self, params, parent=None) -> None:
        super().__init__(parent)
        self.params = params
        self._cancel = False

    def cancel(self) -> None:
        """Request cooperative cancellation; the run loop checks this flag."""
        self._cancel = True

    def run(self) -> None:
        try:
            self._run()
        except Exception as e:
            self.error.emit(str(e))

    def _run(self) -> None:
        b_min = float(self.params["b_min"])
        b_max = float(self.params["b_max"])
        n_nodes = int(self.params["n_nodes"])
        out_path = self.params["output_file"]

        if b_max <= b_min:
            raise ValueError("|B| max must be greater than |B| min.")
        if n_nodes < 2:
            raise ValueError("Node count must be at least 2.")
        if not out_path:
            raise ValueError("Please choose an output file.")

        self._check_li6(self.params)

        if self._cancel:
            self.cancelled.emit()
            return

        self.progress.emit(5)
        constants = ds.li6_d2_constants()

        # generate_table is a single blocking |B| sweep with no progress hook,
        # so cancellation is checked before and after the call (coarse).
        out = ds.generate_table(constants, b_min, b_max, n_nodes, out_path)

        if self._cancel:
            self._delete_partial(out)
            self.cancelled.emit()
            return

        self.progress.emit(100)
        self.finished.emit(out)

    @staticmethod
    def _check_li6(p) -> None:
        """generate_table only reproduces the fixed Li-6 D2 line for now."""
        checks = (
            ("I", p.get("I"), ds.LI6_D2["I"]),
            ("ground J", p.get("ground_J"), ds.LI6_D2["ground"]["J"]),
            ("excited J", p.get("excited_J"), ds.LI6_D2["excited"]["J"]),
        )
        for name, got, want in checks:
            if got is not None and abs(float(got) - float(want)) > 1e-9:
                raise ValueError(
                    f"Only Li-6 D2 tables can be generated currently "
                    f"({name}={want} required, got {got}). General "
                    f"species/line support is deferred (D-05)."
                )

    @staticmethod
    def _delete_partial(path) -> None:
        """Drop an incomplete NPZ so a partial table is never picked up."""
        try:
            os.remove(path)
        except OSError:
            pass
