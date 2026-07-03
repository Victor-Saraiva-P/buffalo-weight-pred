from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from buffalo_weight.csv_io import write_csv_rows


class CsvIoTest(unittest.TestCase):
    def test_write_csv_rows_writes_header_for_empty_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "rows.csv"

            write_csv_rows([], path, ["file_name", "weight"])

            self.assertEqual(path.read_text(), "file_name,weight\n")


if __name__ == "__main__":
    unittest.main()
