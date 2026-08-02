from __future__ import annotations

from pathlib import Path


class FailingSnapshotPublisher:
    def __init__(self) -> None:
        """Create a publication fake.

        Example: ``FailingSnapshotPublisher()`` records attempted publications.
        """
        self.publish_calls = 0

    def publish(self, temporary: Path, destination: Path) -> None:
        """Fail during publication; for example, ``fake.publish(temp, destination)``."""
        self.publish_calls += 1
        raise OSError(
            f"publication destination was {destination}; expected injected publication failure"
        )
