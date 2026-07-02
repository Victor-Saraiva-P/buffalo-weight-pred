from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from buffalo_weight.validation import validate_shared_training_inputs


def write_index(path: Path, file_names: list[str]) -> None:
    strings = ["Nome do arquivo", "Fazenda", "Peso", "Tags", "Fazenda", "train", *file_names]
    rows = [
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>",
        "<worksheet xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'><sheetData>",
        "<row><c t='s'><v>0</v></c><c t='s'><v>1</v></c><c t='s'><v>2</v></c><c t='s'><v>3</v></c></row>",
    ]
    for index, file_name in enumerate(file_names, start=1):
        file_name_index = strings.index(file_name)
        rows.append(
            f"<row><c t='s'><v>{file_name_index}</v></c><c t='s'><v>4</v></c><c><v>{100 + index}</v></c><c t='s'><v>5</v></c></row>"
        )
    rows.append("</sheetData></worksheet>")
    shared_strings = [
        "<?xml version='1.0' encoding='UTF-8' standalone='yes'?>",
        "<sst xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'>",
        *[f"<si><t>{value}</t></si>" for value in strings],
        "</sst>",
    ]
    with ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", "".join(rows))
        archive.writestr("xl/sharedStrings.xml", "".join(shared_strings))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ValidationTest(unittest.TestCase):
    def test_shared_training_inputs_reject_mask_extra_not_in_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks_dir = root / "masks"
            masks_dir.mkdir()
            index_path = root / "indice.xlsx"
            features_path = root / "features.csv"
            split_path = root / "split.csv"
            write_index(index_path, ["mask-001"])
            Image.fromarray(np.asarray([[0, 255]], dtype=np.uint8)).save(masks_dir / "mask-001.png")
            Image.fromarray(np.asarray([[0, 255]], dtype=np.uint8)).save(masks_dir / "mask-extra.png")
            write_csv(features_path, [{"file_name": "mask-001", "weight": "100", "area": "2"}])
            write_csv(
                split_path,
                [
                    {
                        "file_name": "mask-001",
                        "weight": "100",
                        "weight_category": "B1",
                        "weight_category_label": "Faixa 1",
                        "fold": "1",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "Unexpected masks"):
                validate_shared_training_inputs(
                    {
                        "data": {
                            "index_path": str(index_path),
                            "masks_dir": str(masks_dir),
                            "file_name_column": "Nome do arquivo",
                        },
                        "features": {"features_index_path": str(features_path)},
                        "split": {"split_path": str(split_path)},
                    },
                    feature_columns=["area"],
                )

    def test_shared_training_inputs_reject_non_binary_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks_dir = root / "masks"
            masks_dir.mkdir()
            index_path = root / "indice.xlsx"
            features_path = root / "features.csv"
            split_path = root / "split.csv"
            write_index(index_path, ["mask-001"])
            Image.fromarray(np.asarray([[0, 128]], dtype=np.uint8)).save(masks_dir / "mask-001.png")
            write_csv(features_path, [{"file_name": "mask-001", "weight": "100", "area": "2"}])
            write_csv(
                split_path,
                [
                    {
                        "file_name": "mask-001",
                        "weight": "100",
                        "weight_category": "B1",
                        "weight_category_label": "Faixa 1",
                        "fold": "1",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "mask must be binary black/white"):
                validate_shared_training_inputs(
                    {
                        "data": {
                            "index_path": str(index_path),
                            "masks_dir": str(masks_dir),
                            "file_name_column": "Nome do arquivo",
                        },
                        "features": {"features_index_path": str(features_path)},
                        "split": {"split_path": str(split_path)},
                    },
                    feature_columns=["area"],
                )


if __name__ == "__main__":
    unittest.main()
