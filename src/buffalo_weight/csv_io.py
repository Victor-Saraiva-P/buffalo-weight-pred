from __future__ import annotations

import csv
from pathlib import Path


def format_csv_number(value: float) -> str:
    """Format a derived numeric CSV field.

    Example: ``format_csv_number(1.25)`` returns ``"1.250000"``.
    """
    return f"{value:.6f}"


def csv_row_count(path: Path) -> int:
    """Count CSV records after the header.

    Example: ``csv_row_count(Path('folds.csv'))`` counts fold assignments.
    """
    with path.open(encoding="utf-8") as csv_file:
        return max(sum(1 for _ in csv_file) - 1, 0)


def csv_columns(path: Path) -> list[str]:
    """Read an artifact's ordered CSV schema.

    Example: ``csv_columns(Path('folds.csv'))`` returns its header fields.
    """
    with path.open(encoding="utf-8") as csv_file:
        return csv_file.readline().rstrip("\r\n").split(",")


def write_csv_rows(rows: list[dict[str, str]], path: Path, fieldnames: list[str]) -> None:
    """Write CSV rows atomically so partial outputs are not observed.

    Example:
        write_csv_rows([{"fold": "1"}], Path("folds.csv"), ["fold"])
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)
