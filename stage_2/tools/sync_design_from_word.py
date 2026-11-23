#!/usr/bin/env python3
"""Preserve saved Word edits in the existing Markdown without rerendering Word."""
import argparse
from datetime import datetime
import difflib
from pathlib import Path
import re
import shutil
import unicodedata
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def norm(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def markdown_paragraphs(text):
    records = []
    code = False
    for index, line in enumerate(text.splitlines()):
        if line.strip().startswith("```"):
            code = not code
            continue
        if not line.strip():
            continue
        if code:
            value = line
        elif line.startswith(("<!--", "|")):
            continue
        elif line.startswith("!["):
            value = re.fullmatch(r"!\[([^\]]+)\]\(([^)]+)\)", line)[1]
        else:
            value = re.sub(r"^#{1,3} ", "", line)
            refs = bool(re.match(r"^\d+\. ", value) and "http" in value)
            value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                           lambda m: m[1] + ("（" + m[2] + "）" if refs else ""), value)
            value = re.sub(r"`([^`]+)`", r"\1", value).replace("**", "").replace("<br>", "")
        records.append({"line": index, "text": value})
    return records


def word_paragraphs(path):
    records = []
    with zipfile.ZipFile(path) as archive:
        styles = ET.fromstring(archive.read("word/styles.xml"))
        names = {s.get("{" + NS["w"] + "}styleId"): s.find("w:name", NS).get("{" + NS["w"] + "}val")
                 for s in styles.findall("w:style", NS)}
        body = ET.fromstring(archive.read("word/document.xml")).find("w:body", NS)
        for node in body:
            if node.tag != "{" + NS["w"] + "}p":
                continue
            value = "".join(t.text or "" for t in node.findall(".//w:t", NS))
            if not value.strip():
                continue
            style = node.find("w:pPr/w:pStyle", NS)
            name = names.get(style.get("{" + NS["w"] + "}val"), "") if style is not None else ""
            marker = "## " if name.lower() == "heading 1" else "### " if name.lower() == "heading 2" else ""
            records.append({"text": value, "markdown": marker + value})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Back up existing sources, then synchronize Markdown")
    args = parser.parse_args()
    docs = ROOT / "docs"
    source = docs / "设计文档.md"
    old = source.read_text(encoding="utf-8")
    lines = old.splitlines()
    md = markdown_paragraphs(old)
    word = word_paragraphs(docs / "设计文档.docx")
    matcher = difflib.SequenceMatcher(None, [norm(r["text"]) for r in md],
                                      [norm(r["text"]) for r in word], autojunk=False)
    operations = [item for item in matcher.get_opcodes() if item[0] != "equal"]
    for tag, a, b, c, d in reversed(operations):
        begin = md[a]["line"] if a < len(md) else len(lines)
        end = md[b - 1]["line"] + 1 if b > a else begin
        assert not any(line.startswith(("|", "![", "```")) for line in lines[begin:end]), \
            "A structural edit needs manual review; refusing to rewrite tables/images/code blocks"
        replacement = []
        for row in word[c:d]:
            replacement.extend([row["markdown"], ""])
        lines[begin:end] = replacement
    updated = "\n".join(lines).rstrip() + "\n"
    assert [norm(r["text"]) for r in markdown_paragraphs(updated)] == [norm(r["text"]) for r in word]
    diff = "\n".join(difflib.unified_diff(old.splitlines(), updated.splitlines(),
                                         fromfile="previous Markdown", tofile="saved Word text")) + "\n"
    print("Changed paragraph blocks:", len(operations))
    if args.apply and operations:
        history = docs / "history/交付自查前" / datetime.now().strftime("%Y%m%d-%H%M%S")
        history.mkdir(parents=True)
        for name in ["设计文档.md", "设计文档.pdf", "设计文档.html", "设计文档.检查.json", "框架对比摘要核对.json"]:
            if (docs / name).exists():
                shutil.copy2(docs / name, history / name)
        shutil.copy2(docs / "设计文档.docx", history / "设计文档-当前Word保存版.docx")
        (history / "Word修改同步.diff").write_text(diff, encoding="utf-8")
        source.write_text(updated, encoding="utf-8")
        print("Current Word preserved; Markdown updated. Backup:", history)
    elif not args.apply:
        print(diff)


if __name__ == "__main__":
    main()
