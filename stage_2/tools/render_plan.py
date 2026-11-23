#!/usr/bin/env python3
"""Render the project plan as a self-contained, printable HTML document."""

import argparse
from html import escape
from pathlib import Path

from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "plan/项目计划.md")
    args = parser.parse_args()
    source = args.source.resolve()
    output = source.with_suffix(".html")
    content = source.read_text()
    title = next((line[2:].strip() for line in content.splitlines() if line.startswith("# ")), source.stem)
    markdown = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = markdown.parse(content)
    entries = []
    heading_count = 0
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            heading_count += 1
            level = int(token.tag[1:])
            anchor = f"section-{heading_count}"
            token.attrSet("id", anchor)
            if level in (2, 3):
                entries.append((level, anchor, tokens[index + 1].content))
    navigation = "".join(
        f'<a class="level-{level}" href="#{anchor}">{escape(title)}</a>'
        for level, anchor, title in entries
    )
    body = markdown.renderer.render(tokens, markdown.options, {})
    html = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{color-scheme:light;--ink:#1d2939;--muted:#667085;--line:#e4e7ec;--accent:#175cd3}
*{box-sizing:border-box}body{margin:0;background:#f8fafc;color:var(--ink);font:16px/1.85 system-ui,-apple-system,"Noto Sans CJK SC","Microsoft YaHei",sans-serif}
aside{position:fixed;inset:0 auto 0 0;width:285px;background:#fff;border-right:1px solid var(--line);padding:26px 20px;overflow:auto}
aside strong{display:block;margin-bottom:18px;font-size:18px}aside a{display:block;color:#475467;font-size:13px;line-height:1.55;padding:5px 0;text-decoration:none}aside a:hover{color:var(--accent)}aside .level-3{padding-left:12px;font-size:12px}
main{max-width:1120px;margin-left:285px;padding:44px 54px 90px;background:#fff;min-height:100vh}
h1{font-size:29px;line-height:1.45;margin:0 0 22px;letter-spacing:-.5px}h2{font-size:23px;margin:42px 0 14px;padding-bottom:9px;border-bottom:1px solid var(--line)}h3{font-size:18px;margin:27px 0 10px}h1,h2,h3{scroll-margin-top:24px}
p{margin:12px 0}a{color:var(--accent);text-underline-offset:3px}strong{font-weight:650}
table{width:100%;border-collapse:collapse;margin:20px 0;font-size:14px;line-height:1.65}th,td{padding:10px 12px;border:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere}th{background:#f2f4f7;text-align:left}tr:nth-child(even) td{background:#fcfcfd}
pre{background:#f2f4f7;border:1px solid var(--line);border-radius:7px;padding:16px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;line-height:1.65}code{font-family:ui-monospace,"Cascadia Code",monospace;font-size:.88em;overflow-wrap:anywhere}p code,td code,li code{background:#f2f4f7;padding:2px 4px;border-radius:3px}
li{margin:7px 0}ul,ol{padding-left:25px}.stamp{font-size:12px;color:var(--muted);margin-bottom:20px;border-left:3px solid var(--accent);padding-left:12px}
@media(max-width:1050px){aside{display:none}main{margin-left:0;max-width:960px;padding:28px 24px}table{font-size:12px}th,td{padding:7px}}
@media print{@page{size:A4;margin:18mm}body{background:#fff;font-size:10.5pt}aside{display:none}main{margin:0;padding:0;max-width:none}h1{font-size:21pt}h2{font-size:16pt}h3{font-size:12pt}h1,h2,h3{break-after:avoid}tr,pre{break-inside:avoid}table{font-size:9pt}a{color:inherit;text-decoration:none}.stamp{font-size:9pt}}
</style></head><body>
<aside><strong>系统优化项目计划</strong>__NAV__</aside>
<main><div class="stamp">可离线阅读和打印 · 当前主题、实施与验收状态以正文为准</div>__BODY__</main>
</body></html>"""
    output.write_text(html.replace("__TITLE__", escape(title)).replace("__NAV__", navigation).replace("__BODY__", body))
    print(output)


if __name__ == "__main__":
    main()
