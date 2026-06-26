from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError("config root must be a map")
    return loaded
