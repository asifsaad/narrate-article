"""Split a markdown document into chapters.

Shared by every ingest backend, so PDF, EPUB and plaintext all chapterize
identically once they have produced markdown.
"""

from __future__ import annotations

import re

from narrate.document import Chapter

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE)

#: Below this many words, a heading is treated as a subheading of the previous
#: chapter rather than a chapter of its own. Stops PDFs from exploding into
#: hundreds of one-line "chapters" when the parser mislabels running heads or
#: numbered sections.
MIN_CHAPTER_WORDS = 30


def split_into_chapters(
    markdown: str,
    *,
    max_level: int = 2,
    min_words: int = MIN_CHAPTER_WORDS,
    fallback_title: str = "Chapter",
) -> list[Chapter]:
    """Split ``markdown`` at headings of level ``max_level`` or shallower.

    Headings deeper than ``max_level`` stay inline as part of their chapter's
    body, so section structure survives into the narration.

    Chapters shorter than ``min_words`` are merged into the preceding chapter.
    This is the single most important robustness measure for PDFs, where
    heading detection is the known weak spot across every parser.
    """
    matches = [m for m in _HEADING.finditer(markdown) if len(m.group(1)) <= max_level]

    if not matches:
        body = markdown.strip()
        if not body:
            return []
        return [Chapter(index=1, title=fallback_title, markdown=body)]

    sections: list[tuple[str, str]] = []

    # Text before the first heading (front matter, a stray abstract) is not
    # dropped - it is prepended to the first chapter rather than lost.
    preamble = markdown[: matches[0].start()].strip()

    for i, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        sections.append((title, markdown[start:end].strip()))

    if preamble and sections:
        first_title, first_body = sections[0]
        sections[0] = (first_title, f"{preamble}\n\n{first_body}")

    return _merge_short(sections, min_words=min_words)


def _merge_short(sections: list[tuple[str, str]], *, min_words: int) -> list[Chapter]:
    """Fold undersized sections into the previous chapter.

    A short section at the very start has no predecessor, so it absorbs the
    following section instead - that is the "Part One" title-page case.
    """
    merged: list[tuple[str, str]] = []

    for title, body in sections:
        too_short = len(body.split()) < min_words
        if too_short and merged:
            prev_title, prev_body = merged[-1]
            merged[-1] = (prev_title, f"{prev_body}\n\n{body}")
        else:
            merged.append((title, body))

    # A leading short section absorbed nothing above; pull the next one up into
    # it so its title (usually the real part/chapter name) is preserved.
    if len(merged) > 1 and len(merged[0][1].split()) < min_words:
        title, body = merged[0]
        next_title, next_body = merged[1]
        merged[0] = (title, f"{body}\n\n{next_body}")
        del merged[1]

    return [
        Chapter(index=i, title=title, markdown=body)
        for i, (title, body) in enumerate(merged, start=1)
    ]
