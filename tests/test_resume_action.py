"""
Headless tests for the cockpit Resume-run action (Phase 2, plan 02-02).

Verifies the dedicated 'Resume run' action exists on toolbar + Simulation menu and is distinct
from the unpause control (D-02/D-03), that the cockpit slot finds the newest resumable checkpoint
and starts a resume-mode worker (D-02), skips an empty newer batch folder in favour of an older
one that still has a checkpoint (D-08 interaction), and warns without starting a worker when none
exists.

Runs headless (offscreen) and Numba-free: the worker is monkeypatched so .start() never runs a sim.
"""
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication, QMessageBox

import src.batch_worker as batch_worker
import GUI.widgets.features.simulation_cockpit as cockpit_mod
from GUI.main_window import MainWindow

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeSignal:
    def connect(self, *a, **k):
        pass


class _FakeWorker:
    """Records construction args; start() is a no-op (no Numba, no thread)."""
    def __init__(self, directory, file_names, parent=None, buffer_size=10000,
                 checkpoint_interval=30.0, resume_checkpoint_dir=None):
        self.directory = directory
        self.file_names = file_names
        self.resume_checkpoint_dir = resume_checkpoint_dir
        self.started = False
        for s in ("progressChanged", "statusChanged", "fileStarted", "fileFinished", "finished"):
            setattr(self, s, _FakeSignal())

    def start(self):
        self.started = True

    # QThread-like surface the cockpit/closeEvent touches during teardown.
    def isRunning(self):
        return False

    def stop(self):
        pass

    def wait(self, *args):
        pass


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _repo_root_cwd():
    prev = os.getcwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


@pytest.fixture
def window(qapp):
    w = MainWindow(qapp)
    yield w
    w.close()


def _make_checkpoint(results_root, folder_name, with_checkpoint=True):
    run0 = results_root / folder_name / "run_0"
    run0.mkdir(parents=True)
    if with_checkpoint:
        (run0 / "checkpoint.json").write_text("{}")
        (run0 / "checkpoint.npz").write_bytes(b"")
    return run0


def test_resume_run_action_exists_and_distinct(window):
    """The dedicated Resume-run action exists on both bars and is a different object than unpause."""
    tb, mb = window.toolBar, window.menuBar
    assert tb.resume_run_action.text() == "Resume run"
    assert mb.resume_run_action.text() == "Resume run"
    assert tb.resume_run_action is not tb.resume_action   # D-03
    assert mb.resume_run_action is not mb.resume_action


def test_no_checkpoint_does_not_start_worker(window, tmp_path, monkeypatch):
    """With no simulation_results, the slot informs and starts no worker."""
    monkeypatch.setattr(batch_worker, "REPO_ROOT", str(tmp_path))  # no simulation_results dir
    monkeypatch.setattr(cockpit_mod, "BatchSimulationWorker", _FakeWorker)
    calls = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: calls.append(a))
    cockpit = window.simulationCockpitTab
    cockpit.simulation_running_flag = False
    cockpit.batch_worker = None
    cockpit.resume_from_checkpoint()
    assert cockpit.batch_worker is None
    assert calls, "expected an informational 'no checkpoint' message"


def test_checkpoint_found_starts_resume_worker(window, tmp_path, monkeypatch):
    """A resumable checkpoint causes a resume-mode worker to be constructed and started."""
    results_root = tmp_path / "simulation_results"
    run0 = _make_checkpoint(results_root, "01_06_26_0", with_checkpoint=True)
    monkeypatch.setattr(batch_worker, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(cockpit_mod, "BatchSimulationWorker", _FakeWorker)
    cockpit = window.simulationCockpitTab
    cockpit.simulation_running_flag = False
    cockpit.batch_worker = None
    cockpit.resume_from_checkpoint()
    assert isinstance(cockpit.batch_worker, _FakeWorker)
    assert cockpit.batch_worker.started
    assert os.path.normpath(cockpit.batch_worker.resume_checkpoint_dir) == os.path.normpath(str(run0))


def test_older_checkpoint_wins_over_empty_newer(window, tmp_path, monkeypatch):
    """Newest-first scan skips an empty newer batch folder and resumes an older one (D-08 interaction)."""
    results_root = tmp_path / "simulation_results"
    # newer batch folder, NO checkpoint (a later clean run deleted its own, D-08)
    _make_checkpoint(results_root, "02_06_26_0", with_checkpoint=False)
    # older batch folder, still has a resumable checkpoint
    old_run0 = _make_checkpoint(results_root, "01_06_26_0", with_checkpoint=True)
    monkeypatch.setattr(batch_worker, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(cockpit_mod, "BatchSimulationWorker", _FakeWorker)
    cockpit = window.simulationCockpitTab
    cockpit.simulation_running_flag = False
    cockpit.batch_worker = None
    cockpit.resume_from_checkpoint()
    assert isinstance(cockpit.batch_worker, _FakeWorker)
    assert os.path.normpath(cockpit.batch_worker.resume_checkpoint_dir) == os.path.normpath(str(old_run0))
