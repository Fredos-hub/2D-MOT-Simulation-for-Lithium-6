# main.py
import sys
import os
from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QSize, QCoreApplication

from GUI.main_window import MainWindow

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


def main():
    app = QApplication(sys.argv)

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


if __name__ == '__main__':
    main()
