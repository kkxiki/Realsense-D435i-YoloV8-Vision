import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QLibraryInfo
from PyQt5.QtWidgets import QApplication

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

from d435i_camera.config import load_config
from d435i_camera.gui.main_window import MainWindow
from d435i_camera.service import CameraModule


def test_stop_button_path_returns_immediately_and_worker_exits():
    app = QApplication.instance() or QApplication([])
    config = load_config(Path(__file__).parents[1] / "config" / "default.yaml")
    config.camera.backend = "mock"
    module = CameraModule(config)
    window = MainWindow(module)
    window.start_stream()
    deadline = time.monotonic() + 3.0
    while module.latest is None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert module.latest is not None

    started = time.monotonic()
    window.stop_stream()
    elapsed = time.monotonic() - started
    assert elapsed < 0.25

    deadline = time.monotonic() + 3.0
    while window.worker and window.worker.isRunning() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert window.worker is not None and not window.worker.isRunning()

    # Reproduce the settings workflow: wait for the old worker, then immediately
    # start a new one. Queued "camera stopped" signals from the old worker must not
    # disable the new worker's Stop button.
    old_worker = window.worker
    window.start_stream()
    new_worker = window.worker
    assert new_worker is not old_worker
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert new_worker is not None and new_worker.isRunning()
    assert window.stop_button.isEnabled()
    assert not window.start_button.isEnabled()

    window.stop_stream(wait=True)
    window.close()
