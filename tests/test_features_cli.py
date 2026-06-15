from __future__ import annotations

import os
import csv
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from zipfile import ZipFile


def write_minimal_xlsx(path: Path, rows: list[list[str]]) -> None:
    shared: list[str] = []
    shared_index: dict[str, int] = {}

    def cell_ref(column_index: int, row_index: int) -> str:
        return f"{chr(ord('A') + column_index)}{row_index}"

    def shared_id(value: str) -> int:
        if value not in shared_index:
            shared_index[value] = len(shared)
            shared.append(value)
        return shared_index[value]

    sheet_rows = []
    for row_index, row in enumerate(rows, start=1):
        cells = []
        for column_index, value in enumerate(row):
            cells.append(
                f'<c r="{cell_ref(column_index, row_index)}" t="s"><v>{shared_id(value)}</v></c>'
            )
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    shared_items = "".join(f"<si><t>{value}</t></si>" for value in shared)
    sheet_data = "".join(sheet_rows)

    with ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
<Override PartName="/xl/sharedStrings.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"/>
</Types>""",
        )
        archive.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings" Target="sharedStrings.xml"/>
</Relationships>""",
        )
        archive.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Planilha1" sheetId="1" r:id="rId1"/></sheets>
</workbook>""",
        )
        archive.writestr(
            "xl/sharedStrings.xml",
            f"""<?xml version="1.0" encoding="UTF-8"?>
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="{len(shared)}" uniqueCount="{len(shared)}">{shared_items}</sst>""",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f"""<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{sheet_data}</sheetData></worksheet>""",
        )


def write_grayscale_png(path: Path, pixels: list[list[int]]) -> None:
    height = len(pixels)
    width = len(pixels[0])

    def chunk(kind: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)

    raw = b"".join(b"\x00" + bytes(row) for row in pixels)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk("IHDR".encode(), struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk("IDAT".encode(), zlib.compress(raw))
        + chunk("IEND".encode(), b"")
    )


class FeatureCliTest(unittest.TestCase):
    def test_fails_with_missing_mask_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            masks_dir = data_dir / "conjunto-de-mascaras"
            masks_dir.mkdir(parents=True)
            index_path = data_dir / "indice.xlsx"
            write_minimal_xlsx(
                index_path,
                [
                    ["Nome do arquivo", "Fazenda", "Peso", "Tags"],
                    ["existing", "Manezinho", "110", "ok"],
                    ["missing", "Manezinho", "120", "ok"],
                ],
            )
            (masks_dir / "existing.png").write_bytes(b"not decoded in this test")
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "name: test_features",
                        "data:",
                        "  index_path: data/indice.xlsx",
                        "  masks_dir: data/conjunto-de-mascaras",
                        "  file_name_column: Nome do arquivo",
                        "  farm_column: Fazenda",
                        "  target_column: Peso",
                        "  tag_column: Tags",
                        "features:",
                        "  include:",
                        "    - area",
                        "output:",
                        "  features_index_path: generated/features.csv",
                        "  columns:",
                        "    - file_name",
                        "    - farm",
                        "    - weight",
                        "    - tag",
                        "    - area",
                    ]
                )
            )

            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path.cwd() / "src")
            result = subprocess.run(
                [sys.executable, "-m", "buffalo_weight.features", "--config", str(config_path)],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing masks", result.stderr)
            self.assertIn("missing", result.stderr)

    def test_generates_feature_index_with_configured_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            masks_dir = data_dir / "conjunto-de-mascaras"
            masks_dir.mkdir(parents=True)
            write_minimal_xlsx(
                data_dir / "indice.xlsx",
                [
                    ["Nome do arquivo", "Fazenda", "Peso", "Tags"],
                    ["mask-1", "Manezinho", "110", "ok"],
                ],
            )
            write_grayscale_png(
                masks_dir / "mask-1.png",
                [
                    [0, 0, 0],
                    [0, 255, 255],
                    [0, 255, 255],
                ],
            )
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "name: test_features",
                        "data:",
                        "  index_path: data/indice.xlsx",
                        "  masks_dir: data/conjunto-de-mascaras",
                        "  file_name_column: Nome do arquivo",
                        "  farm_column: Fazenda",
                        "  target_column: Peso",
                        "  tag_column: Tags",
                        "features:",
                        "  include:",
                        "    - area",
                        "output:",
                        "  features_index_path: generated/features.csv",
                        "  columns:",
                        "    - file_name",
                        "    - farm",
                        "    - weight",
                        "    - tag",
                        "    - area",
                    ]
                )
            )

            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path.cwd() / "src")
            result = subprocess.run(
                [sys.executable, "-m", "buffalo_weight.features", "--config", str(config_path)],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            with (root / "generated/features.csv").open(newline="") as file:
                reader = csv.DictReader(file)
                rows = list(reader)

            self.assertEqual(
                reader.fieldnames,
                ["file_name", "farm", "weight", "tag", "area"],
            )
            self.assertEqual(rows[0]["file_name"], "mask-1")
            self.assertEqual(rows[0]["farm"], "Manezinho")
            self.assertEqual(rows[0]["weight"], "110")
            self.assertEqual(rows[0]["tag"], "ok")
            self.assertEqual(rows[0]["area"], "4")


if __name__ == "__main__":
    unittest.main()
