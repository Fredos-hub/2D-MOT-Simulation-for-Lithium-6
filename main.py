# main.py
import sys
import os
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QSize, QCoreApplication
from argparse import ArgumentParser
from src.batch_worker import BatchSimulationWorker
from GUI.main_window import MainWindow
import qdarktheme
import traceback
from tqdm import tqdm
# Enable per-monitor DPI scaling before creating the application
QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
QCoreApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

# Define a baseline resolution for scaling (e.g. 1920x1080)
BASE_WIDTH = 1920
BASE_HEIGHT = 1080




def fade_out_splash(splash, main_window):
    fade = QPropertyAnimation(splash, b"windowOpacity")
    fade.setDuration(1000)
    fade.setStartValue(1.0)
    fade.setEndValue(0.0)
    fade.finished.connect(lambda: (splash.close(), main_window.show()))
    fade.start()
    splash.animation = fade  # keep reference alive

def start_gui(qt_argv,args):

    app = QApplication(qt_argv)
    if args.style == "light":
        app.setStyleSheet(qdarktheme.load_stylesheet("light"))
    elif args.style == "dark":
        app.setStyleSheet(qdarktheme.load_stylesheet("dark"))



    # Store baseline resolution for later reference
    app.setProperty('baseResolution', (BASE_WIDTH, BASE_HEIGHT))

    # Load and scale the pixmap up to 90% of current screen
    icon_path = os.path.join(os.path.dirname(__file__), "GUI/icons/simulation_logo_5.png")
    pixmap = QPixmap(icon_path)
    screen = app.primaryScreen()
    screen_geom = screen.availableGeometry()
    max_w, max_h = int(screen_geom.width() * 0.9), int(screen_geom.height() * 0.9)
    if pixmap.width() > max_w or pixmap.height() > max_h:
        pixmap = pixmap.scaled(QSize(max_w, max_h), Qt.KeepAspectRatio, Qt.SmoothTransformation)

    # Show splash immediately
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.setMask(pixmap.mask())
    splash.setWindowOpacity(0.0)
    splash.show()
    app.processEvents()

    # Fade-in animation
    fade_in = QPropertyAnimation(splash, b"windowOpacity")
    fade_in.setDuration(1000)
    fade_in.setStartValue(0.0)
    fade_in.setEndValue(1.0)
    def on_fade_in_finished():
        # Construct main window after fade-in
        main_window = MainWindow(app)
        QTimer.singleShot(2000, lambda: fade_out_splash(splash, main_window))
    fade_in.finished.connect(on_fade_in_finished)
    fade_in.start()
    splash.animation = fade_in

    sys.exit(app.exec_())


def on_progress_changed(pbar, p):
    delta = p - pbar.n
    if delta > 0:
        pbar.update(delta)

def main():

    parser = ArgumentParser(description="Run the simulation GUI")

    parser.add_argument("-s", "--style", type=str, default="light", choices=["light", "dark"], help="Set the application style (default: light)")
    parser.add_argument("--GUI", action="store_true", help="Run the GUI application")
    parser.add_argument("--files", nargs='*', help="List of files to run")
    parser.add_argument("--target-dir", type=str, default="2D-MOT-Simulation-For-Lithium-6", help="Target directory for simulation outputs")
    args, unknown = parser.parse_known_args()
    qt_argv = [sys.argv[0]] + unknown
    
    if args.GUI:
        start_gui(qt_argv,args)
    else:
        # Basic validation / user feedback
        if not args.files:
            print("No files provided. Use --files file1.json file2.json or run with --GUI.")
            return
        
        progress_bar = tqdm(
                            total=100,
                            desc="Simulation Progress",
                            unit="%",
                            position=0,
                            leave=True,
                            dynamic_ncols=True
                        )
        
        paths_to_files = [os.path.join(os.getcwd(), f) for f in args.files]
        worker = BatchSimulationWorker(args.target_dir, paths_to_files)

        # connect signals to console output so we can see status in CLI mode
        worker.statusChanged.connect(lambda s: tqdm.write(f"[STATUS] {s}"))
        worker.progressChanged.connect(lambda p: on_progress_changed(progress_bar, p)) # returns progress in percent (0-100)
        worker.fileFinished.connect(lambda f: tqdm.write(f"[FILE DONE] {f}"))
        worker.finished.connect(lambda: tqdm.write("[FINISHED] All batch jobs completed."))

        # run synchronously (worker.run() executes in current thread)
        try:
            worker.run()
        except Exception:
            # print full traceback so we get useful debug info in CLI mode
            print("Exception while running batch worker:")
            traceback.print_exc()






if __name__ == '__main__':
    main()
