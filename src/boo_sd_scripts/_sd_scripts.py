"""Puts the sd-scripts checkout (git submodule) on sys.path.

sd-scripts is not installable as a library, so extensions import it from its directory. Import this module before
importing `train_network`, `moss_image_train_network` or `library.*`.
"""

import os
import sys
from pathlib import Path


def sd_scripts_path() -> Path:
    override = os.environ.get("SD_SCRIPTS_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "sd-scripts"


def ensure_on_path() -> Path:
    path = sd_scripts_path()
    if not (path / "train_network.py").is_file():
        raise ImportError(
            f"train_network.py not found in {path}. Run `git submodule update --init` or set SD_SCRIPTS_PATH to an sd-scripts checkout."
        )
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


ensure_on_path()
