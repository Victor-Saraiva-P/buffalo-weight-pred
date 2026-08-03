"""Filesystem boundary for atomic reconstructed-stage publication."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Protocol


class AtomicFilesystem(Protocol):
    """Minimal atomic I/O seam; for example, tests can fail the second replacement."""

    def replace(self, source: Path, destination: Path) -> None:
        """Replace one path.

        Example: ``filesystem.replace(next_link, current)`` swaps pointers.
        """
        ...

    def symlink(self, target: str, link: Path) -> None:
        """Create a pointer.

        Example: ``filesystem.symlink(target, next_link)`` prepares publication.
        """
        ...

    def remove_tree(self, path: Path) -> None:
        """Remove obsolete storage.

        Example: ``filesystem.remove_tree(old_snapshot)`` reclaims a cache.
        """
        ...

    def unlink(self, path: Path) -> None:
        """Remove one pointer.

        Example: ``filesystem.unlink(next_link)`` cleans an abandoned swap.
        """
        ...


class SystemAtomicFilesystem:
    """System I/O adapter; for example, it powers ``FilesystemSnapshotPublisher``."""

    def replace(self, source: Path, destination: Path) -> None:
        """Replace one path.

        Example: ``filesystem.replace(next_link, current)`` swaps pointers.
        """
        os.replace(source, destination)

    def symlink(self, target: str, link: Path) -> None:
        """Create a pointer.

        Example: ``filesystem.symlink(target, next_link)`` prepares publication.
        """
        os.symlink(target, link)

    def remove_tree(self, path: Path) -> None:
        """Remove obsolete storage.

        Example: ``filesystem.remove_tree(old_snapshot)`` reclaims a cache.
        """
        shutil.rmtree(path, ignore_errors=True)

    def unlink(self, path: Path) -> None:
        """Remove one pointer.

        Example: ``filesystem.unlink(next_link)`` cleans an abandoned swap.
        """
        path.unlink(missing_ok=True)


class SnapshotPublisher(Protocol):
    """Publication seam; for example, a fake can reject ``publish`` atomically."""

    def publish(self, temporary: Path, destination: Path) -> None:
        """Publish a complete snapshot.

        Example: ``publisher.publish(temp, final)`` installs validated outputs.
        """
        ...


class FilesystemSnapshotPublisher:
    """Atomic pointer publisher; for example, readers always see one complete snapshot."""

    def __init__(self, filesystem: AtomicFilesystem | None = None) -> None:
        """Inject atomic I/O.

        Example: ``FilesystemSnapshotPublisher(fake_filesystem)`` tests interrupted swaps.
        """
        self._filesystem = filesystem or SystemAtomicFilesystem()

    def publish(self, temporary: Path, destination: Path) -> None:
        """Swap one pointer; for example, ``publisher.publish(temp, current)`` is atomic."""
        _reject_legacy_destination(destination)
        previous = _pointed_snapshot(destination)
        snapshot_store = destination.parent / ".snapshots" / destination.name
        snapshot_store.mkdir(parents=True, exist_ok=True)
        installed = snapshot_store / temporary.name
        self._filesystem.replace(temporary, installed)
        next_link = destination.with_name(f".{destination.name}-next-{temporary.name}")
        self._install_pointer(installed, next_link, destination)
        if previous is not None:
            self._filesystem.remove_tree(previous)

    def _install_pointer(self, snapshot: Path, next_link: Path, destination: Path) -> None:
        relative_target = os.path.relpath(snapshot, destination.parent)
        self._filesystem.symlink(relative_target, next_link)
        try:
            self._filesystem.replace(next_link, destination)
        except BaseException:
            self._filesystem.unlink(next_link)
            raise


def clean_snapshot_stage(
    destination: Path, filesystem: AtomicFilesystem | None = None
) -> None:
    """Remove storage; for example, ``clean_snapshot_stage(inputs_path, filesystem)``."""
    resolved_filesystem = filesystem or SystemAtomicFilesystem()
    if destination.is_symlink():
        resolved_filesystem.unlink(destination)
    elif destination.is_dir():
        resolved_filesystem.remove_tree(destination)
    snapshot_store = destination.parent / ".snapshots" / destination.name
    resolved_filesystem.remove_tree(snapshot_store)


def _reject_legacy_destination(destination: Path) -> None:
    if destination.exists() and not destination.is_symlink():
        raise ValueError(
            f"stage destination was {destination}; expected absent path or managed symlink"
        )


def _pointed_snapshot(destination: Path) -> Path | None:
    if not destination.is_symlink():
        return None
    pointed = destination.resolve(strict=False)
    expected_store = (destination.parent / ".snapshots" / destination.name).resolve()
    try:
        pointed.relative_to(expected_store)
    except ValueError as error:
        raise ValueError(
            f"stage pointer target was {pointed}; expected a descendant of {expected_store}"
        ) from error
    return pointed
