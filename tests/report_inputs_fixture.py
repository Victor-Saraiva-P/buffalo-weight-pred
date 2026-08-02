from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


class CuratedInputsFixture:
    """Build a temporary curated sample; for example, ``CuratedInputsFixture(root, 50)``."""

    def __init__(self, root: Path, sample_count: int = 50) -> None:
        self.root = root
        self.sample_count = sample_count
        self.index_path = root / "data" / "mask_index.csv"
        self.masks_dir = root / "data" / "masks"
        self.config_path = root / "report.json"
        self.output_dir = root / "generated" / "report" / "inputs"
        self.masks_dir.mkdir(parents=True)
        self._write_index()
        self._write_masks()
        self._write_config()

    def run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Run the real CLI; for example, ``fixture.run('inputs', '--dry-run')``."""
        main_path = Path(__file__).parents[1] / "main.py"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
        return subprocess.run(
            [sys.executable, str(main_path), *arguments, "--config", str(self.config_path)],
            cwd=self.root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def replace_weight(self, row_index: int, weight: str) -> None:
        """Change one label; for example, ``fixture.replace_weight(0, '81')``."""
        rows = self.index_rows()
        rows[row_index]["weight_kg"] = weight
        self._write_csv(rows)

    def replace_file_name(self, row_index: int, file_name: str) -> None:
        """Change one key; for example, ``fixture.replace_file_name(1, 'mask-000.png')``."""
        rows = self.index_rows()
        rows[row_index]["file_name"] = file_name
        self._write_csv(rows)

    def index_rows(self) -> list[dict[str, str]]:
        """Read fixture labels; for example, ``fixture.index_rows()[0]``."""
        with self.index_path.open(newline="") as index_file:
            return list(csv.DictReader(index_file))

    def mask_path(self, row_index: int) -> Path:
        """Locate one mask.

        Example: ``fixture.mask_path(0)`` returns the first fixture PNG.
        """
        return self.masks_dir / f"mask-{row_index:03d}.png"

    def _write_index(self) -> None:
        rows = [
            {
                "file_name": f"mask-{index:03d}.png",
                "farm": "Faco" if index % 2 else "Manezinho",
                "weight_kg": str(80 + index * 3),
            }
            for index in range(self.sample_count)
        ]
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_csv(rows)

    def _write_csv(self, rows: list[dict[str, str]]) -> None:
        with self.index_path.open("w", newline="") as index_file:
            writer = csv.DictWriter(index_file, fieldnames=["file_name", "farm", "weight_kg"])
            writer.writeheader()
            writer.writerows(rows)

    def _write_masks(self) -> None:
        for index in range(self.sample_count):
            height = 8 + index // 10
            width = 8 + index % 10
            pixels = np.zeros((height, width), dtype=np.uint8)
            pixels[1:-1, 1 : 2 + index % (width - 2)] = 255
            pixels[1 + index % (height - 2), 1:-1] = 255
            Image.fromarray(pixels).save(self.mask_path(index))

    def _write_config(self) -> None:
        contract = {
            "inputs": {
                "mask_index_path": str(self.index_path),
                "masks_dir": str(self.masks_dir),
                "expected_mask_count": self.sample_count,
                "canonical_long_side": 1024,
                "weight_category_count": 10,
                "fold_count": 5,
                "fold_seed": 42,
            },
            "artifacts": {"root": str(self.root / "generated" / "report")},
        }
        self.config_path.write_text(json.dumps(contract))
