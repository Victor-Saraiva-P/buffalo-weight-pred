from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from buffalo_weight.feature_selection_io import load_feature_samples
from buffalo_weight.input_schema import FEATURE_COLUMNS, SPLIT_COLUMNS


class FeatureSelectionIoTest(unittest.TestCase):
    def test_invalid_feature_reports_field_and_offending_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inputs_dir = Path(directory)
            feature_row = {field: "1" for field in FEATURE_COLUMNS}
            feature_row.update({"file_name": "mask.png", "farm": "Faco", "area": "not-a-number"})
            split_row = {field: "1" for field in SPLIT_COLUMNS}
            split_row.update({"file_name": "mask.png", "farm": "Faco", "weight_category": "B1"})
            _write_selection_input_csv(inputs_dir / "feature_index.csv", FEATURE_COLUMNS, feature_row)
            _write_selection_input_csv(inputs_dir / "canonical_split.csv", SPLIT_COLUMNS, split_row)

            with self.assertRaisesRegex(ValueError, "area.*'not-a-number'.*finite numeric"):
                load_feature_samples(inputs_dir, ("area",))


def _write_selection_input_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    unittest.main()
