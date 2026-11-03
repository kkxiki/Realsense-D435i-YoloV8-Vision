"""Command-line diagnostics and headless capture entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .config import load_config
from .service import CameraModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="XuK D435i camera module")
    parser.add_argument("--config", default=None, help="YAML configuration file")
    parser.add_argument("--mock", action="store_true", help="use synthetic RGB-D input")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("info", help="list camera devices and module health")
    capture = subparsers.add_parser("capture", help="process frames and save the last snapshot")
    capture.add_argument("--frames", type=int, default=10)
    capture.add_argument("--output", default=None)
    run = subparsers.add_parser("run", help="run headlessly and print latest observations")
    run.add_argument("--frames", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.mock:
        config.camera.backend = "mock"
        config.camera.serial = ""
    module = CameraModule(config)
    try:
        if args.command == "info":
            devices = [vars(item) for item in module.list_devices()]
            print(json.dumps({"devices": devices, "health": module.health()}, ensure_ascii=False, indent=2))
            return 0
        module.start()
        result = None
        for _ in range(max(1, args.frames)):
            result = module.process_once()
        assert result is not None
        if args.command == "capture":
            paths = module.capture_snapshot(args.output)
            print(
                json.dumps(
                    {
                        "color": str(paths.color),
                        "depth": str(paths.depth) if paths.depth else None,
                        "metadata": str(paths.metadata),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        module.close()

