"""Markdown and plain-text ingest.

A file that is already markdown has no Docling label tree, so this is the one
place where structural exclusion has to be textual. The job is the one
``labels.py`` does for PDFs: keep the prose, drop everything else before the
normalizer sees it.

Hand-written markdown - notes, blog posts, READMEs - carries furniture that
Docling output never does: front matter, HTML comments, footnote definitions,
task lists, callouts. It is all handled here rather than in the normalizer,
which also serves the Docling path and should not need to know about any of it.
"""

from __future__ import annotations

import datetime
import html
import re
from pathlib import Path

from narrate.document import Book
from narrate.ingest.chapterize import split_into_chapters

__all__ = ["clean_markdown", "ingest_markdown", "split_front_matter"]


def ingest_markdown(path: Path, *, split_level: int = 2) -> Book:
    """Read a .md or .txt file into a :class:`Book` of speakable markdown."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    front, body = split_front_matter(raw)
    text = clean_markdown(body)

    front_title = _scalar(front.get("title"))
    title = front_title or _first_h1(text) or _title_from_stem(path)

    # Blog engines keep the title in front matter only. Without this the
    # listener never hears what the piece is called.
    if front_title and not _first_h1(text):
        text = f"# {front_title}\n\n{text}"

    metadata = {str(key): _scalar(value) for key, value in front.items()}
    chapters = split_into_chapters(text, max_level=split_level, fallback_title=title)
    return Book(
        title=title,
        author=_scalar(front.get("author") or front.get("authors")),
        chapters=chapters,
        source_path=path,
        metadata={key: value for key, value in metadata.items() if value},
    )


# --- front matter ---------------------------------------------------------

_YAML_FRONT = re.compile(r"\A---[ \t]*\n(.*?)^(?:---|\.\.\.)[ \t]*$\n?", re.MULTILINE | re.DOTALL)
_TOML_FRONT = re.compile(r"\A\+\+\+[ \t]*\n(.*?)^\+\+\+[ \t]*$\n?", re.MULTILINE | re.DOTALL)


def split_front_matter(text: str) -> tuple[dict, str]:
    """Separate a leading YAML (``---``) or TOML (``+++``) block from the body.

    Read aloud, front matter is "title colon why sleep matters tags colon..."
    before the first real word. A leading block that does not parse as a
    mapping is left alone - it is a horizontal rule, not metadata.
    """
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")

    for pattern, parse in ((_YAML_FRONT, _parse_yaml), (_TOML_FRONT, _parse_toml)):
        match = pattern.match(text)
        if not match:
            continue
        data = parse(match.group(1))
        if data is None:
            data = {}
        if isinstance(data, dict):
            return data, text[match.end():]

    return {}, text


def _parse_yaml(block: str):
    import yaml

    try:
        return yaml.safe_load(block)
    except yaml.YAMLError:
        return False


def _parse_toml(block: str):
    import tomllib

    try:
        return tomllib.loads(block)
    except tomllib.TOMLDecodeError:
        return False


def _scalar(value) -> str | None:
    """Flatten a front-matter value to display text. Mappings have none."""
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float, datetime.date)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        parts = [_scalar(item) for item in value]
        return ", ".join(part for part in parts if part) or None
    return None


# --- body -----------------------------------------------------------------

_FENCE = re.compile(
    r"^[ \t]{0,3}(`{3,}|~{3,}).*?(?:^[ \t]{0,3}\1[`~]*[ \t]*$|\Z)",
    re.MULTILINE | re.DOTALL,
)
_COMMENT = re.compile(r"<!--.*?-->|%%.*?%%", re.DOTALL)
_SCRIPT_OR_STYLE = re.compile(r"<(script|style)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_MATH_BLOCK = re.compile(r"\$\$.+?\$\$", re.DOTALL)

# A footnote definition runs to the next blank line, plus any indented
# paragraphs after it.
_FOOTNOTE_DEF = re.compile(
    r"^[ \t]{0,3}\[\^[^\]\n]+\]:[^\n]*(?:\n[^\n]+)*(?:\n\n(?: {4}|\t)[^\n]*(?:\n[^\n]+)*)*",
    re.MULTILINE,
)
_FOOTNOTE_REF = re.compile(r"\[\^[^\]\s]+\]")
# The destination must be a single token with an optional quoted title, so a
# prose line such as "[Note]: this matters" is not mistaken for a definition.
_LINK_DEF = re.compile(
    r"^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*<?[^\s>]+>?"
    r"(?:[ \t]+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^)\n]*\)))?[ \t]*$\n?",
    re.MULTILINE,
)

_EMBED = re.compile(r"!\[\[[^\]\n]*\]\]")
_IMAGE = re.compile(r"!\[[^\]]*\](?:\([^)]*\)|\[[^\]]*\])")
_WIKILINK = re.compile(r"\[\[([^\]|#\n]*)(?:#[^\]|\n]*)?(?:\|([^\]\n]*))?\]\]")
_REF_LINK = re.compile(r"\[([^\]\n]+)\]\[[^\]\n]*\]")
_AUTOLINK = re.compile(r"<(?:(?:https?|ftp|mailto):[^>\s]+|[^@<>\s]+@[^@<>\s]+)>")

_TASK_BOX = re.compile(r"^([ \t]*(?:>[ \t]?)*(?:[-*+]|\d+[.)])[ \t]+)\[[ xX]\][ \t]+", re.MULTILINE)
_CALLOUT = re.compile(r"^([ \t]*>[ \t]?)\[!([A-Za-z-]+)\][+-]?[ \t]*(.*)$", re.MULTILINE)

_LINE_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
# Inline code is matched first and kept, so "the `<details>` tag" survives.
_HTML_TAG_OUTSIDE_CODE = re.compile(r"(`[^`\n]+`)|</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")
_INLINE_MARKERS = re.compile(r"(~~|==)(?=\S)(.+?)(?<=\S)\1")
_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~<=$])")

_TABLE_ROW = re.compile(r"^[ \t]*\|.*\|[ \t]*$")
_TABLE_DIVIDER = re.compile(r"^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*$")
_SETEXT_UNDERLINE = re.compile(r"^[ \t]{0,3}(=+|-+)[ \t]*$")
_NOT_PARAGRAPH = re.compile(r"^[ \t]*(?:#|>|\||[-*+][ \t]|\d+[.)][ \t])")


def clean_markdown(text: str) -> str:
    """Reduce hand-written markdown to the subset the normalizer understands.

    Ordering matters. Code and comments go first because they can contain
    anything, including syntax the later passes would act on. Definitions go
    before references, so a footnote's body is removed rather than left behind
    with its marker stripped.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = _FENCE.sub("", text)
    text = _COMMENT.sub("", text)
    text = _SCRIPT_OR_STYLE.sub("", text)
    text = _MATH_BLOCK.sub("", text)
    text = _drop_tables(text)
    text = _setext_to_atx(text)

    text = _FOOTNOTE_DEF.sub("", text)
    text = _LINK_DEF.sub("", text)
    text = _FOOTNOTE_REF.sub("", text)

    text = _EMBED.sub("", text)
    text = _IMAGE.sub("", text)
    text = _WIKILINK.sub(lambda m: (m.group(2) or m.group(1)).strip(), text)
    text = _REF_LINK.sub(r"\1", text)
    text = _AUTOLINK.sub("", text)

    text = _TASK_BOX.sub(r"\1", text)
    text = _CALLOUT.sub(_speak_callout, text)

    text = _LINE_BREAK.sub("\n", text)
    text = _HTML_TAG_OUTSIDE_CODE.sub(lambda m: m.group(1) or "", text)
    text = _INLINE_MARKERS.sub(r"\2", text)
    text = _ESCAPE.sub(r"\1", text)
    text = html.unescape(text)

    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _drop_tables(text: str) -> str:
    """Remove pipe tables, with or without outer pipes.

    A row bounded by pipes is always dropped. A divider row (``---|:--:``)
    also takes the header above it and the body rows below, which is what
    catches tables written without the outer pipes.
    """
    lines = text.split("\n")
    keep = [True] * len(lines)

    for i, line in enumerate(lines):
        if _TABLE_ROW.match(line):
            keep[i] = False
        elif _TABLE_DIVIDER.match(line):
            keep[i] = False
            if i and "|" in lines[i - 1]:
                keep[i - 1] = False
            j = i + 1
            while j < len(lines) and lines[j].strip() and "|" in lines[j]:
                keep[j] = False
                j += 1

    return "\n".join(line for line, kept in zip(lines, keep) if kept)


def _speak_callout(match: re.Match[str]) -> str:
    """``> [!WARNING] Hot`` -> ``> Hot.``; ``> [!NOTE]`` -> ``> Note.``

    A custom title replaces the type name, as it does when rendered.
    """
    prefix, kind, title = match.groups()
    label = title.strip() or kind.replace("-", " ").capitalize()
    if label[-1] not in ".!?:;":
        label += "."
    return f"{prefix}{label}"


def _setext_to_atx(text: str) -> str:
    """Rewrite ``Title`` over ``=====`` as ``# Title``, so it splits chapters.

    Only a one-line paragraph is converted. A ``---`` under anything else is a
    horizontal rule, and a stray ``===`` line is dropped as noise.
    """
    out: list[str] = []

    for line in text.split("\n"):
        underline = _SETEXT_UNDERLINE.match(line)
        if underline:
            previous = out[-1] if out else ""
            starts_paragraph = len(out) < 2 or not out[-2].strip()
            if previous.strip() and starts_paragraph and not _NOT_PARAGRAPH.match(previous):
                level = "#" if underline.group(1).startswith("=") else "##"
                out[-1] = f"{level} {previous.strip()}"
                continue
            if underline.group(1).startswith("="):
                continue
        out.append(line)

    return "\n".join(out)


def _first_h1(text: str) -> str | None:
    heading = re.search(r"^#[ \t]+(.+?)[ \t]*$", text, re.MULTILINE)
    return heading.group(1).strip() if heading else None


def _title_from_stem(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip() or "Untitled"
