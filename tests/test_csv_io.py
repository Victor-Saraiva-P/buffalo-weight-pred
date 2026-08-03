from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from buffalo_weight.csv_io import csv_columns, csv_row_count, format_csv_number, write_csv_rows


class CsvIoTest(unittest.TestCase):
    def test_reads_schema_count_and_formats_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            path.write_text("file_name,weight_kg\nmask.png,110.000000\n")

            self.assertEqual(csv_columns(path), ["file_name", "weight_kg"])
            self.assertEqual(csv_row_count(path), 1)
            self.assertEqual(format_csv_number(1.25), "1.250000")

    def test_write_csv_rows_writes_header_for_empty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "rows.csv"

            write_csv_rows([], path, ["file_name", "weight"])

            self.assertEqual(path.read_text(), "file_name,weight\n")


if __name__ == "__main__":
    unittest.main()
