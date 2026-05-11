import gc
import os
import re
import json
import time
from datetime import datetime
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from src.parameters import Parameters, ParameterError
from util.simulation_typing import ECSAtoms

class BatchSimulationWorker(QThread):
    progressChanged = pyqtSignal(int)
    statusChanged = pyqtSignal(str)
    fileFinished = pyqtSignal(str)
    finished = pyqtSignal()
    fileStarted = pyqtSignal(str, int)    # filename, total_steps


    def __init__(self, directory: str, file_names: list, parent=None, buffer_size: int = 10000):
        super().__init__(parent)
        self.directory = directory
        self.file_names = file_names
        self._pause = False
        self._stop = False
        self._stop_current = False   # cancel only the current file, then continue

        # --- file state ---
        self.batch_root = None          # path to simulation_results
        self.batch_folder = None        # path to DD_MM_YY_NUM
        self.run_fhs = {}               # run_idx -> open file handle for result.csv
        self.run_header_written = {}    # run_idx -> bool
        self.run_write_options = {}     # run_idx -> dict of chosen write options
        # buffering:
        self.run_buffers = {}           # run_idx -> list[str] (buffered CSV lines)
        self.run_buffer_size = buffer_size  # flush threshold


    # -----------------------
    # filesystem helper funcs
    # -----------------------
    def ensure_batch_root_and_folder(self):
        """
        Ensure `simulation_results` exists one level above self.directory,
        and create a dated batch folder named DD_MM_YY_NUM (NUM auto-increments).
        """
        workspace = os.path.abspath(os.path.join(self.directory, ".."))
        batch_root = os.path.join(workspace, "simulation_results")
        os.makedirs(batch_root, exist_ok=True)

        today = datetime.now().strftime("%d_%m_%y")  # DD_MM_YY
        pattern = re.compile(rf'^{re.escape(today)}_(\d+)$')

        maxnum = -1
        for name in os.listdir(batch_root):
            m = pattern.match(name)
            if m:
                try:
                    n = int(m.group(1))
                    if n > maxnum:
                        maxnum = n
                except ValueError:
                    pass

        newnum = (maxnum + 1) if maxnum >= 0 else 0
        batch_folder_name = f"{today}_{newnum}"
        batch_folder_path = os.path.join(batch_root, batch_folder_name)
        os.makedirs(batch_folder_path, exist_ok=False)
        self.batch_root = batch_root
        self.batch_folder = batch_folder_path
        self.statusChanged.emit(f"Created batch folder: {self.batch_folder}")

    def make_run_folder(self, idx: int):
        """
        Create run_{idx} folder inside batch folder and open result.csv (no header yet).
        """
        if self.batch_folder is None:
            raise RuntimeError("batch folder not created")

        run_folder = os.path.join(self.batch_folder, f"run_{idx}")
        os.makedirs(run_folder, exist_ok=True)

        csv_path = os.path.join(run_folder, "result.csv")
        # Open in append mode so we don't overwrite if re-running same idx inadvertently
        fh = open(csv_path, "a", newline="")
        self.run_fhs[idx] = fh
        self.run_header_written[idx] = False
        self.run_write_options[idx] = None
        self.run_buffers[idx] = []  # initialize buffer for this run
        self.statusChanged.emit(f"Run folder ready: {run_folder}")
        return run_folder

    def flush_run_buffer(self, idx: int):
        """
        Flush the buffer for run idx to disk.
        """
        fh = self.run_fhs.get(idx)
        buf = self.run_buffers.get(idx)
        if fh is None or buf is None:
            return
        if len(buf) == 0:
            return
        try:
            fh.write("".join(buf))
            fh.flush()
        except Exception as e:
            # non-fatal; emit status for debugging
            self.statusChanged.emit(f"Error flushing buffer for run {idx}: {e}")
        finally:
            # clear buffer even if write partially failed to avoid duplicates
            self.run_buffers[idx] = []

    def close_run(self, idx: int):
        # Flush before closing
        try:
            self.flush_run_buffer(idx)
        except Exception:
            pass

        fh = self.run_fhs.get(idx)
        if fh:
            try:
                fh.close()
            except Exception:
                pass
            if idx in self.run_fhs:
                del self.run_fhs[idx]
        if idx in self.run_header_written:
            del self.run_header_written[idx]
        if idx in self.run_write_options:
            del self.run_write_options[idx]
        if idx in self.run_buffers:
            del self.run_buffers[idx]

    def _make_header_from_opts(self, opts: dict):
        cols = ["step", "atom_id"]
        if opts.get("write_position", True):
            cols += ["position_x", "position_y", "position_z"]
        if opts.get("write_velocity", True):
            cols += ["velocity_x", "velocity_y", "velocity_z"]
        if opts.get("write_subjective_time", True):
            cols += ["subjective_time"]
        if opts.get("write_excitation_count", True):
            cols += ["excitation_count"]
        if opts.get("write_ground_state", True):
            cols += ["current_groundstate"]
        return ",".join(cols) + "\n"

    def write_step_results(self, run_idx: int, step: int, current_atom_states: ECSAtoms = None, alive_ids=None,
                        excitation_counter=None,
                        write_position: bool = True,
                        write_velocity: bool = True,
                        write_subjective_time: bool = True,
                        write_excitation_count: bool = False,
                        write_ground_state: bool = False,
                        force_write_all: bool = False):
        """
        Simpler, cleaner implementation that writes only alive atoms using direct
        boolean/integer indexing into the per-atom arrays stored on current_atom_states.

        Assumptions:
        - current_atom_states has per-atom numpy arrays named exactly as used below
            (positions, velocities, subjective_time, status, groundstates, atom_ids, etc).
        - `alive_ids` (when provided) is a numpy array of integer indices (e.g. output of check_if_alive).
        - Minimal error handling: if these assumptions are violated an exception will be raised.
        """
        fh = self.run_fhs.get(run_idx)
        if fh is None:
            raise RuntimeError(f"No open result file for run {run_idx}")

        opts = {
            "write_position": bool(write_position),
            "write_velocity": bool(write_velocity),
            "write_subjective_time": bool(write_subjective_time),
            "write_excitation_count": bool(write_excitation_count),
            "write_ground_state": bool(write_ground_state),
        }

        # Header (written once)
        if not self.run_header_written.get(run_idx, False):
            header = self._make_header_from_opts(opts)
            fh.write(header)
            fh.flush()
            self.run_header_written[run_idx] = True
            self.run_write_options[run_idx] = opts
        else:
            opts = self.run_write_options[run_idx]

        # Prefer explicit excitation_counter, otherwise try to use one from current_atom_states
        exc = excitation_counter if excitation_counter is not None else getattr(current_atom_states, "excitation_counter", None)

        # Determine alive indices (absolute indices into per-atom arrays)
        if force_write_all:
            n = int(getattr(current_atom_states, "n"))
            alive_idx = np.arange(n, dtype=int)
        else:
            if alive_ids is None:
                status = np.asarray(current_atom_states.status)
                alive_idx = np.nonzero(status == 1)[0]
            else:
                a = np.asarray(alive_ids)
                alive_idx = a.astype(int) if a.dtype != bool else np.nonzero(a)[0]

        # Slice per-atom arrays for the alive atoms 
        # (these will have length == alive_idx.size)
        positions = current_atom_states.positions[alive_idx] if opts["write_position"] else None
        velocities = current_atom_states.velocities[alive_idx] if opts["write_velocity"] else None
        subjective_time = current_atom_states.subjective_time[alive_idx] if opts["write_subjective_time"] else None
        groundstates = current_atom_states.groundstates[alive_idx] if opts["write_ground_state"] else None

        exc_alive = None
        if opts["write_excitation_count"]:
            exc_alive = np.asarray(exc)[alive_idx] if exc is not None else None

        # Build buffer lines for alive atoms only
        buf = self.run_buffers.get(run_idx) or []
        for i, atom_id in enumerate(alive_idx):
            parts = [str(int(step)), str(int(atom_id))]

            if opts["write_position"]:
                p = positions[i]                  # shape (3,)
                parts += [str(p[0]), str(p[1]), str(p[2])]

            if opts["write_velocity"]:
                v = velocities[i]                 # shape (3,)
                parts += [str(v[0]), str(v[1]), str(v[2])]

            if opts["write_subjective_time"]:
                parts.append(f"{subjective_time[i]:.8f}")

            if opts["write_excitation_count"]:
                parts.append(str(int(exc_alive[i]) if exc_alive is not None else 0))

            if opts["write_ground_state"]:
                # groundstates may be 1D or 2D; convert to scalar if needed
                g = groundstates[i]
                if np.ndim(g) == 0:
                    parts.append(str(int(g)))
                else:
                    # if it's a row (e.g. shape (2,)), join by '|' or pick first column depending on desired format
                    # here we join with '|' to represent multi-component ground state compactly
                    parts.append("|".join(str(int(x)) for x in np.ravel(g)))

            buf.append(",".join(parts) + "\n")

        self.run_buffers[run_idx] = buf

        if len(buf) >= self.run_buffer_size:
            self.flush_run_buffer(run_idx)

    # -----------------------
    # main run loop (modified)
    # -----------------------
    def run(self):
        total_files = len(self.file_names)

        try:
            # create batch folder only once, before processing simulations
            self.ensure_batch_root_and_folder()
        except Exception as e:
            self.statusChanged.emit(f"Failed to create batch folder: {e}")
            self.batch_folder = None

        for idx, filename in enumerate(self.file_names):
            if self._stop:
                break

            # create run folder (each simulation run gets run_{idx})
            try:
                run_folder = self.make_run_folder(idx)
            except Exception as e:
                self.statusChanged.emit(f"Failed to create run folder for {filename}: {e}")
                run_folder = None

            # 1) Build simulation


            self.statusChanged.emit(f"---------------Building {filename} ({idx+1}/{total_files})------------------")

            params = Parameters(os.path.join(self.directory, filename), status_callback=self.statusChanged.emit)
            self.fileStarted.emit(filename, params.max_step_number)
            # TODO: Consider moving to loading of the files.
            if not params.is_valid():
                # single GUI message referencing the errors and then full listing once
                self.statusChanged.emit(f"Configuration invalid: {len(params.get_errors())} error(s). See details below.")
                self.statusChanged.emit("---- Validation errors ----\n" + "\n".join(params.get_errors()))
                # optionally save the invalid config for inspection
                if run_folder is not None and isinstance(params.parameters, dict):
                    try:
                        cfg_path = os.path.join(run_folder, "config_invalid.json")
                        with open(cfg_path, "w", encoding="utf-8") as cfgfh:
                            json.dump(params.parameters, cfgfh, indent=2)
                    except Exception:
                        pass
                self.fileFinished.emit(filename)
                continue

            # params is valid, attempt to build simulation once
            try:
                sim = params.build_simulation()
            except ParameterError as exc:
                # build_simulation reports runtime/fatal construction issues here
                msg = f"Failed to build simulation: {exc}"
                # include parsing warnings if any
                if params.get_errors():
                    msg += "\nWarnings:\n" + "\n".join(params.get_errors())
                self.statusChanged.emit(msg)
                self.fileFinished.emit(filename)
                continue

            # at this point sim exists and is safe to use
            self.statusChanged.emit("Compiling... (this may take a couple of minutes)")
            try:
                sim.warmup()
            except Exception as e:
                self.statusChanged.emit(f"Warmup failed: {e}")
                # ensure final cleanup
                try:
                    sim.finalize()
                except Exception:
                    pass
                self.fileFinished.emit(filename)
                continue

            self.statusChanged.emit("Starting simulation...")

            total_steps = sim.max_step_number
            start_time = time.perf_counter()
            last_update_time = start_time
            last_update_iter = sim.current_step

            try:
                # 3) Run simulation steps
                for i in range(sim.current_step, total_steps):
                    if self._stop:
                        pct = int(i / total_steps * 100) if total_steps > 0 else 0
                        self.statusChanged.emit(
                            f"Cancelled {filename} at step {i}/{total_steps} ({pct}%) "
                            f"— partial results saved. Stopping all."
                        )
                        break
                    if self._stop_current:
                        pct = int(i / total_steps * 100) if total_steps > 0 else 0
                        self.statusChanged.emit(
                            f"Cancelled {filename} at step {i}/{total_steps} ({pct}%) "
                            f"— partial results saved. Continuing with remaining files."
                        )
                        break
                    while self._pause:
                        self.msleep(100)

                    # inside run() for each step i:
                    cont, current_atom_states, returned_excitation_counter, alive_idx, exc_hist = sim.step(i)
                    
                    # prefer the returned excitation counter (falls back to sim attr)
                    excitation_counter = returned_excitation_counter if returned_excitation_counter is not None else getattr(sim, "excitation_counter", None)

                    # write (buffered)
                    if alive_idx.size > 0 and run_folder is not None:
                        try:
                            self.write_step_results(
                                run_idx=idx,
                                step=i,
                                current_atom_states=current_atom_states,
                                alive_ids=alive_idx,
                                excitation_counter=excitation_counter,
                                write_position=True,
                                write_velocity=True,
                                write_subjective_time=True,
                                write_excitation_count=True,
                                write_ground_state=True
                            )
                        except Exception as e:
                            self.statusChanged.emit(f"Error writing step results (run {idx} step {i}): {e}")

                    progress = int((i+1)/total_steps*100)
                    self.progressChanged.emit(progress)

                    # Estimated time update
                    now = time.perf_counter()
                    if now - last_update_time >= 1.0:
                        iters = (i+1) - last_update_iter
                        if iters > 0:
                            avg = (now - last_update_time)/iters
                            rem = int(avg*(total_steps - (i+1)))
                            self.statusChanged.emit(
                                f"Processing step {i+1}/{total_steps}... (est. {rem}s)"
                            )
                        last_update_time = now
                        last_update_iter = i+1

                    if not cont:
                        print(exc_hist)
                        self.statusChanged.emit("Simulation ended early: no atoms alive.")
                        self.progressChanged.emit(100)
                        break

                # Ensure final flush / close of this run
                if run_folder is not None:
                    # write config.json next to result.csv using the original parsed JSON
                    try:
                        cfg_path = os.path.join(run_folder, "config.json")
                        # Prefer to save the original parsed dictionary for compactness
                        if hasattr(params, "parameters") and isinstance(params.parameters, dict):
                            with open(cfg_path, "w", encoding="utf-8") as cfgfh:
                                json.dump(params.parameters, cfgfh, indent=2)
                        else:
                            # fallback to a minimal export of attributes
                            minimal = {
                                "max_step_number": getattr(params, "max_step_number", None),
                                "step_resolution": getattr(params, "step_resolution", None),
                                "simulated_time": getattr(params, "simulated_time", None),
                                "atom_number": getattr(params, "atom_number", None),
                                "lasers": getattr(params, "lasers", None),
                                "magnetic_field_type": getattr(params, "magnetic_field_type", None)
                            }
                            with open(cfg_path, "w", encoding="utf-8") as cfgfh:
                                json.dump(minimal, cfgfh, indent=2)
                    except Exception as e:
                        self.statusChanged.emit(f"Failed to write config.json for {filename}: {e}")

            except Exception as e:
                self.statusChanged.emit(f"Exception during simulation ({filename}): {e}")
            finally:
                try:
                    self.close_run(idx)
                except Exception:
                    pass
                sim.finalize()
                duration = time.perf_counter() - start_time
                self.statusChanged.emit(f"----------------Completed {filename} in {duration:.2f}s.----------------")
                self.fileFinished.emit(filename)
                # Reset per-file cancel flag and release memory
                self._stop_current = False
                del sim
                del params
                gc.collect()

        # Batch finished
        self.finished.emit()

    def pause(self):
        self._pause = True

    def resume(self):
        self._pause = False

    def stop_current(self):
        """Cancel the running simulation, then continue with remaining files."""
        self._stop_current = True
        self._pause = False  # unblock if currently paused

    def stop(self):
        """Cancel all remaining simulations."""
        self._stop = True
        self._stop_current = True  # also unblock the inner loop immediately
        self._pause = False        # unblock if currently paused
