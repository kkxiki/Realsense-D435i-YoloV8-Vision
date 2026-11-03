"""GUI entry point."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from PyQt5.QtCore import QLibraryInfo, QTimer, Qt
from PyQt5.QtWidgets import QApplication

from ..config import load_config
from ..service import CameraModule
from .main_window import MainWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XuK D435i camera GUI")
    parser.add_argument("--config", default=None, help="YAML configuration file")
    parser.add_argument("--mock", action="store_true", help="use the built-in RGB-D simulator")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.mock:
        config.camera.backend = "mock"
        config.camera.serial = ""
        config.detector.model_path = ""
        # Keep simulator UI changes out of the real D435i configuration. Several GUI
        # actions persist the whole config, so a mock session must use a disposable file.
        config.config_path = Path(tempfile.gettempdir()) / "xuk_d435i_mock_runtime.yaml"
    # opencv-python bundles a different Qt plugin tree and may overwrite this
    # environment variable during import. Mixing that xcb plugin with PyQt5 causes
    # an immediate crash on the Ubuntu desktop, so explicitly select PyQt5's tree.
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
        QLibraryInfo.PluginsPath
    )
    app = QApplication(sys.argv[:1])
    app.setApplicationName("XuK D435i Camera")
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app.setStyle("Fusion")
    module = CameraModule(config)
    window = MainWindow(module)
    window.show()
    QTimer.singleShot(0, window.start_stream)
    return int(app.exec_())


if __name__ == "__main__":
    raise SystemExit(main())
