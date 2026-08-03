"""Shared hashing primitives for artifact provenance."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Hash one file in bounded memory.

    Example: ``sha256_file(Path('artifact.csv'))`` returns its hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
