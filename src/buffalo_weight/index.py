from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_index(path: Path) -> list[dict[str, str]]:
    with ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", NS):
                shared_strings.append(
                    "".join(
                        text.text or ""
                        for text in item.iter(
                            "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
                        )
                    )
                )

        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        raw_rows = []
        for row in sheet_root.findall("a:sheetData/a:row", NS):
            values = []
            for cell in row.findall("a:c", NS):
                value = cell.find("a:v", NS)
                text = "" if value is None else value.text or ""
                if cell.attrib.get("t") == "s" and text:
                    text = shared_strings[int(text)]
                values.append(text)
            raw_rows.append(values)

    if not raw_rows:
        return []

    headers = raw_rows[0]
    return [
        {headers[index]: value for index, value in enumerate(row) if index < len(headers)}
        for row in raw_rows[1:]
    ]
