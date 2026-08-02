"""Filesystem boundary for atomic reconstructed-stage publication."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Protocol


class SnapshotPublisher(Protocol):
    """Publication seam; for example, a fake can reject ``publish`` atomically."""

    def publish(self, temporary: Path, destination: Path) -> None:
        """Publish a complete snapshot.

        Example: ``publisher.publish(temp, final)`` installs validated outputs.
        """
        ...


class FilesystemSnapshotPublisher:
    """Filesystem publisher; for example, it swaps a validated temporary stage."""

    def publish(self, temporary: Path, destination: Path) -> None:
        """Atomically replace a snapshot; for example, ``publisher.publish(temp, final)``."""
        backup = destination.with_name(f".{destination.name}-previous")
        self._remove_previous_backup(backup)
        if destination.exists():
            os.replace(destination, backup)
        self._install_or_restore(temporary, destination, backup)

    @staticmethod
    def _remove_previous_backup(backup: Path) -> None:
        if not backup.exists():
            return
        shutil.rmtree(backup)

    @staticmethod
    def _install_or_restore(temporary: Path, destination: Path, backup: Path) -> None:
        try:
            os.replace(temporary, destination)
        except BaseException:
            if backup.exists():
                os.replace(backup, destination)
            raise
        shutil.rmtree(backup, ignore_errors=True)
