"""Git boundary for publishing confirmed feature evidence."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol


class FeatureConfirmationEnvironment(Protocol):
    """Expose repository state; for example, tests inject a named clean environment."""

    def worktree_changes(self, repository_root: Path) -> tuple[str, ...]:
        """Return porcelain entries.

        Example: an empty tuple permits promotion.
        """
        ...


class LocalFeatureConfirmationEnvironment:
    """Read local Git state; for example, promotion rejects modified tracked files."""

    def worktree_changes(self, repository_root: Path) -> tuple[str, ...]:
        """Return porcelain entries; for example, untracked review files remain visible."""
        result = subprocess.run(
            ["git", "-C", str(repository_root), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        )
        return tuple(line for line in result.stdout.splitlines() if line)
