from __future__ import annotations

import os
import shutil
from pathlib import Path


class FailAfterSnapshotInstallOperations:
    def __init__(self) -> None:
        """Create an atomic-filesystem fake.

        Example: ``FailAfterSnapshotInstallOperations()`` fails after the first rename.
        """
        self.replace_calls = 0

    def replace(self, source: Path, destination: Path) -> None:
        """Install then fail; for example, the second ``replace`` raises ``OSError``."""
        self.replace_calls += 1
        if self.replace_calls == 2:
            raise OSError(f"destination was {destination}; expected post-install failure")
        os.replace(source, destination)

    def symlink(self, target: str, link: Path) -> None:
        """Create a real pointer.

        Example: ``fake.symlink(target, link)`` prepares the failed swap.
        """
        os.symlink(target, link)

    def remove_tree(self, path: Path) -> None:
        """Remove obsolete storage.

        Example: ``fake.remove_tree(snapshot)`` records realistic cleanup.
        """
        shutil.rmtree(path, ignore_errors=True)

    def unlink(self, path: Path) -> None:
        """Remove a pointer; for example, ``fake.unlink(next_link)`` cleans failure state."""
        path.unlink(missing_ok=True)
