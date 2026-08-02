from __future__ import annotations

import io
from pathlib import PurePosixPath


class RecordingBinaryFile(io.BytesIO):
    def __init__(self, path: "MemoryPath") -> None:
        """Record writes into the owning in-memory path.

        Example: closing this stream makes its bytes readable from ``path``.
        """
        super().__init__()
        self.path = path

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        if exception_type is None:
            self.path.files[self.path.value] = self.getvalue()
        self.close()


class MemoryPath:
    def __init__(self, value: str, files: dict[str, bytes] | None = None) -> None:
        """Create a fake path; for example, seed ``files`` with cached bytes."""
        self.value = value
        self.files = files if files is not None else {}

    @property
    def parent(self) -> "MemoryPath":
        """Return a shared-store parent; for example, ``a/b`` returns ``a``."""
        return MemoryPath(str(PurePosixPath(self.value).parent), self.files)

    @property
    def suffix(self) -> str:
        """Return the fake suffix; for example, ``weights.pth`` returns ``.pth``."""
        return PurePosixPath(self.value).suffix

    def exists(self) -> bool:
        """Report stored content; for example, seeded paths exist immediately."""
        return self.value in self.files

    def mkdir(self, parents: bool, exist_ok: bool) -> None:
        """Accept directory creation; for example, setup can prepare a fake parent."""
        return None

    def with_suffix(self, suffix: str) -> "MemoryPath":
        """Return a sibling fake path; for example, append the setup ``.part`` suffix."""
        return MemoryPath(str(PurePosixPath(self.value).with_suffix(suffix)), self.files)

    def replace(self, target: "MemoryPath") -> None:
        """Promote stored bytes; for example, replace a cache with its ``.part`` file."""
        self.files[target.value] = self.files.pop(self.value)

    def open(self, mode: str) -> io.BytesIO:
        if mode == "rb":
            return io.BytesIO(self.files[self.value])
        if mode == "wb":
            return RecordingBinaryFile(self)
        raise ValueError(f"memory file mode was {mode!r}; expected 'rb' or 'wb'")

    def write_text(self, content: str) -> int:
        encoded = content.encode()
        self.files[self.value] = encoded
        return len(encoded)

    def unlink(self, missing_ok: bool) -> None:
        """Remove fake content; for example, clean a failed partial download."""
        if not missing_ok or self.value in self.files:
            self.files.pop(self.value)

    def __str__(self) -> str:
        """Render the fake path; for example, include it in validation errors."""
        return self.value
