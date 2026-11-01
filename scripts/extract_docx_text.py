"""Extract paragraphs and tables from an OOXML document in document order.

This intentionally uses only the Python standard library so the official IEEE
source documents can be audited without installing a document converter.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def text(node: ET.Element) -> str:
    # Word text uses w:t while Office Math uses m:t.  Keeping both is vital for
    # parameter formulas embedded in IEEE tables.
    return "".join(item.text or "" for item in node.iter()
                   if item.tag.rsplit("}", 1)[-1] == "t").strip()


def extract(source: Path) -> list[str]:
    with zipfile.ZipFile(source) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    body = root.find(f"{W}body")
    if body is None:
        raise ValueError(f"No Word document body in {source}")
    lines: list[str] = []
    for child in body:
        if child.tag == f"{W}p":
            value = text(child)
            if value:
                lines.append(value)
        elif child.tag == f"{W}tbl":
            lines.append("[TABLE]")
            for row in child.findall(f"{W}tr"):
                cells = [text(cell).replace("\t", " ") for cell in row.findall(f"{W}tc")]
                lines.append("\t".join(cells))
            lines.append("[/TABLE]")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text("\n".join(extract(args.source)) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
