from __future__ import annotations

from pathlib import Path


def load_config(path: Path) -> dict[str, object]:
    root: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object] | list[str]]] = [(-1, root)]

    for raw_line in path.read_text().splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()

        parent = stack[-1][1]
        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Invalid list item in config: {raw_line}")
            parent.append(line[2:].strip().strip('"'))
            continue

        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip().strip('"')

        if value:
            if not isinstance(parent, dict):
                raise ValueError(f"Invalid scalar in config: {raw_line}")
            parent[key] = value
            continue

        container: dict[str, object] | list[str]
        container = [] if key in {"include", "columns", "feature_columns", "models"} else {}
        if not isinstance(parent, dict):
            raise ValueError(f"Invalid section in config: {raw_line}")
        parent[key] = container
        stack.append((indent, container))

    return root
