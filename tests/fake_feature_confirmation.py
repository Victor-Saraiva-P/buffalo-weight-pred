from __future__ import annotations

from pathlib import Path


class FixedFeatureConfirmationEnvironment:
    """Expose deterministic Git state; for example, tests can report one dirty path."""

    def __init__(self, changed_paths: tuple[str, ...] = ()) -> None:
        """Store visible changes; for example, ``('.dirty',)`` blocks promotion."""
        self.changed_paths = changed_paths
        self.checked_roots: list[Path] = []

    def worktree_changes(self, repository_root: Path) -> tuple[str, ...]:
        """Return injected changes; for example, a clean fake returns an empty tuple."""
        self.checked_roots.append(repository_root)
        return self.changed_paths
