#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_DIR"
"$PYTHON_BIN" -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools>=68,<80' wheel
python -m pip install -e . --no-deps

python - <<'PY'
import importlib.util
import subprocess
import sys

required = {
    "pyrealsense2": "pyrealsense2>=2.55",
    "yaml": "PyYAML>=6.0",
    "PIL": "Pillow>=9.0",
}
missing = [package for module, package in required.items() if importlib.util.find_spec(module) is None]
try:
    import lzma  # noqa: F401
except ModuleNotFoundError:
    if importlib.util.find_spec("backports.lzma") is None:
        missing.append("backports.lzma>=0.0.14")
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

optional = ["cv2", "PyQt5", "ultralytics"]
not_found = [name for name in optional if importlib.util.find_spec(name) is None]
if not_found:
    print("Optional modules not found:", ", ".join(not_found))
    print("Install all optional dependencies with: python -m pip install -e '.[all]'")
PY

mkdir -p data/captures data/calibration data/logs models
echo "Installation complete. Run: scripts/run_gui.sh"
