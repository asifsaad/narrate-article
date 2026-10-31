"""The display side: what the reader sees, and how it divides into blocks.

Display text is the speakable Markdown that ingest produced (what ``extract``
would show), with tables and code already gone. The page renders it as prose,
so two things are derived from it here:

- **blocks** - headings, paragraphs, list items and quotes, which become the
  page's elements and the units "previous/next paragraph" steps through;
- **markup** - character spans that are Markdown syntax (``#``, ``**``, ``- ``,
  a link's ``](url)``), which the page hides. Content the narrator skips but a
  reader should see - a citation, a URL in prose, a lexicon drop - is *not*
  markup; it stays visible and is simply never highlighted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BLOCK_KINDS = ("heading", "paragraph", "list_item", "quote")

_HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+\S")
_HRULE = re.compile(r"^[ \t]*(?:[-*_][ \t]*){3,}$")
_LIST_ITEM = re.compile(r"^[ \t]*(?:[-*+]|(\d+)[.)])[ \t]+\S")
_QUOTE = re.compile(r"^[ \t]*>")

# Used with ``pattern.match(text, pos, endpos)``, which anchors at ``pos`` by
# itself. A leading ``^`` would not: without MULTILINE it only matches at the
# real start of the string, so every block after the first lost its markers.
_HEADING_MARKS = re.compile(r"( {0,3}#{1,6}[ \t]+)(?:.*?)([ \t]+#+[ \t]*)?$")
_LIST_MARK = re.compile(r"[ \t]*(?:[-*+]|\d+[.)])[ \t]+")
_QUOTE_MARK = re.compile(r"[ \t]*(?:>[ \t]?)+")

_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"(\[)[^\]\n]+(\]\([^)\n]*\))")
_STAR_EMPHASIS = re.compile(r"(\*{1,3})(?=\S)(?:(?!\n[ \t]*\n).)+?(?<=\S)\1", re.DOTALL)
_UNDERSCORE_EMPHASIS = re.compile(
    r"(?<!\w)(_{1,3})(?=\S)(?:(?!\n[ \t]*\n).)+?(?<=\S)\1(?!\w)", re.DOTALL
)


@dataclass(frozen=True)
class Block:
    id: int
    kind: str
    src: tuple[int, int]
    """The whole block in the display text, Markdown markers included."""
    level: int | None = None
    """Heading level, 1-6."""
    number: int | None = None
    """The number of an ordered list item; ``None`` for a bullet."""

    def to_json(self) -> dict:
        data: dict = {"id": self.id, "kind": self.kind, "src": list(self.src)}
        if self.level is not None:
            data["level"] = self.level
        if self.number is not None:
            data["number"] = self.number
        return data


def parse_blocks(text: str) -> list[Block]:
    """Split display Markdown into blocks, line by line.

    A blank line ends a block. A heading is always one line. A list item runs
    until a blank line or the next item, so a wrapped item stays one block. A
    paragraph line straight after a list item is its continuation, as in
    CommonMark's lazy continuation.
    """
    blocks: list[Block] = []
    current: dict | None = None

    def close() -> None:
        nonlocal current
        if current is not None:
            blocks.append(
                Block(
                    id=len(blocks),
                    kind=current["kind"],
                    src=(current["start"], current["end"]),
                    level=current.get("level"),
                    number=current.get("number"),
                )
            )
        current = None

    offset = 0
    for line in text.split("\n"):
        start, end = offset, offset + len(line)
        offset = end + 1

        if not line.strip() or _HRULE.match(line):
            close()
            continue

        if heading := _HEADING.match(line):
            close()
            current = {"kind": "heading", "start": start, "end": end, "level": len(heading.group(1))}
            close()
            continue

        if item := _LIST_ITEM.match(line):
            close()
            number = int(item.group(1)) if item.group(1) else None
            current = {"kind": "list_item", "start": start, "end": end, "number": number}
            continue

        if _QUOTE.match(line):
            if current is None or current["kind"] != "quote":
                close()
                current = {"kind": "quote", "start": start, "end": end}
            else:
                current["end"] = end
            continue

        if current is None:
            current = {"kind": "paragraph", "start": start, "end": end}
        else:
            current["end"] = end

    close()
    return blocks


def markup_spans(text: str, blocks: list[Block]) -> list[tuple[int, int]]:
    """Character spans of Markdown syntax, sorted and non-overlapping."""
    spans: list[tuple[int, int]] = []

    offset = 0
    for line in text.split("\n"):
        start = offset
        offset += len(line) + 1
        if line.strip() and _HRULE.match(line):
            spans.append((start, start + len(line)))

    for block in blocks:
        a, b = block.src
        first_line_end = text.find("\n", a, b)
        first_line_end = b if first_line_end < 0 else first_line_end
        if block.kind == "heading":
            match = _HEADING_MARKS.match(text, a, b)
            if match:
                spans.append(match.span(1))
                if match.group(2):
                    spans.append(match.span(2))
        elif block.kind == "list_item":
            match = _LIST_MARK.match(text, a, first_line_end)
            if match:
                spans.append(match.span())
        elif block.kind == "quote":
            line_start = a
            while line_start < b:
                line_end = text.find("\n", line_start, b)
                line_end = b if line_end < 0 else line_end
                match = _QUOTE_MARK.match(text, line_start, line_end)
                if match:
                    spans.append(match.span())
                line_start = line_end + 1

    code: list[tuple[int, int]] = []
    for match in _INLINE_CODE.finditer(text):
        code.append(match.span())
        spans.append((match.start(), match.start() + 1))
        spans.append((match.end() - 1, match.end()))

    def outside_code(position: int) -> bool:
        return not any(a <= position < b for a, b in code)

    for match in _LINK.finditer(text):
        if outside_code(match.start()):
            spans.append(match.span(1))
            spans.append(match.span(2))
    for pattern in (_STAR_EMPHASIS, _UNDERSCORE_EMPHASIS):
        for match in pattern.finditer(text):
            if outside_code(match.start()):
                marker = len(match.group(1))
                spans.append((match.start(), match.start() + marker))
                spans.append((match.end() - marker, match.end()))

    return _merge(spans)


def visible_mask(length: int, markup: list[tuple[int, int]]) -> list[bool]:
    """``True`` for each display character a reader sees."""
    mask = [True] * length
    for a, b in markup:
        for i in range(max(a, 0), min(b, length)):
            mask[i] = False
    return mask


def block_at(blocks: list[Block], position: int) -> Block | None:
    """The block containing ``position``, or the nearest one before it."""
    found = None
    for block in blocks:
        if block.src[0] > position:
            break
        found = block
    return found


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for a, b in sorted(span for span in spans if span[1] > span[0]):
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


__all__ = ["BLOCK_KINDS", "Block", "block_at", "markup_spans", "parse_blocks", "visible_mask"]
