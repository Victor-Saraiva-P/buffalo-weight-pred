from __future__ import annotations

import csv
from pathlib import Path


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
