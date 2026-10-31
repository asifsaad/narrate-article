"""Tie engine tokens to the spoken text, then to the display text.

Two steps, kept separate because only the second is temporary:

1. :func:`locate_tokens` finds each token in its chunk's *spoken* text with a
   moving cursor. That is exact and permanent.
2. :func:`naive_display_spans` maps spoken words onto the *display* text by
   aligning the two word sequences. It stands in until the normalizer keeps
   receipts (Phase 4): a word that survives normalization unchanged ("deal")
   is matched directly; words the normalizer produced ("four point two million
   dollars") are not in the display text at all, so they share the display
   span between the matched words either side - which for ``$4.2M`` and
   ``Dr.`` happens to be right, and for a run of several rewrites in a row
   ("1939-1945, e.g.") is one coarse unit.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher

#: 1:1 replacements, so folded text keeps the original's character offsets.
_FOLD = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "\n": " ", "\t": " ", "\r": " "})

_NEXT_WORD = re.compile(r"\S+")

#: A display word: letters and digits, with inner apostrophes or hyphens.
_DISPLAY_WORD = re.compile(r"\w+(?:['’-]\w+)*")

#: Trimmed from the edges of a gap before unmatched words share it: spacing,
#: list bullets and punctuation that belongs to the sentence, not the word.
#: A trailing full stop is kept, so "Dr." and "e.g." stay whole.
_GAP_LEFT = " \t\n,;:!?.)]}*_#>-\"'“”‘’"
_GAP_RIGHT = " \t\n,;:!?)]}(*_\"'“”‘’"


def is_word(text: str) -> bool:
    """A token a reader would see highlighted: it has a letter or digit."""
    return any(char.isalnum() for char in text)


def fold(text: str) -> str:
    """Case- and quote-insensitive form with the same length as ``text``."""
    return "".join(
        lower if len(lower := char.lower()) == 1 else char for char in text.translate(_FOLD)
    )


def locate_tokens(
    text: str, tokens: Sequence[str]
) -> tuple[list[tuple[int, int] | None], int]:
    """Each token's ``[start, end)`` in ``text``, and how many words missed.

    Tokens arrive in order, so the search starts where the previous one ended
    and may skip only whitespace and punctuation. A word token that is not
    found there takes the next whitespace-delimited word instead and counts as
    a miss; a punctuation token that is not found is simply left unplaced.
    """
    folded = fold(text)
    spans: list[tuple[int, int] | None] = []
    misses = 0
    cursor = 0

    for token in tokens:
        needle = fold(token).strip()
        if not needle:
            spans.append(None)
            continue

        position = folded.find(needle, cursor)
        if position >= 0 and not is_word(folded[cursor:position]):
            spans.append((position, position + len(needle)))
            cursor = position + len(needle)
            continue

        if not is_word(token):
            spans.append(None)
            continue

        misses += 1
        following = _NEXT_WORD.search(text, cursor)
        if following is None:
            spans.append(None)
            continue
        spans.append(following.span())
        cursor = following.end()

    return spans, misses


def word_core(token: str) -> str:
    """The token without leading or trailing punctuation: ``F.`` -> ``F``."""
    start, end = 0, len(token)
    while start < end and not token[start].isalnum():
        start += 1
    while end > start and not token[end - 1].isalnum():
        end -= 1
    return token[start:end]


def naive_display_spans(
    display: str,
    visible: Sequence[bool],
    words: Sequence[str],
    *,
    start: int,
    end: int,
) -> tuple[list[tuple[int, int]], int]:
    """Display spans for ``words`` within ``display[start:end]``, and how many
    had to share a gap rather than match directly.

    The spoken words and the visible display words are aligned as two
    sequences (case- and quote-insensitive), which keeps matches in order and
    cannot be derailed by a common word ("for") turning up further on: taking
    a distant match would cost every match in between. Spoken words left
    unmatched between two matches share the trimmed display text between
    them, which makes them one highlight unit; if that gap is empty they share
    the previous match instead.
    """
    shown = [
        match
        for match in _DISPLAY_WORD.finditer(display, start, end)
        if all(visible[match.start() : match.end()])
    ]
    spoken = [fold(word_core(word)) for word in words]
    matcher = SequenceMatcher(None, spoken, [fold(m.group()) for m in shown], autojunk=False)

    anchors: list[tuple[int, int] | None] = [None] * len(words)
    for spoken_at, shown_at, size in matcher.get_matching_blocks():
        for k in range(size):
            if spoken[spoken_at + k]:
                anchors[spoken_at + k] = shown[shown_at + k].span()

    spans: list[tuple[int, int]] = []
    shared = 0
    previous_end = start
    index = 0
    while index < len(anchors):
        if anchors[index] is not None:
            spans.append(anchors[index])
            previous_end = anchors[index][1]
            index += 1
            continue

        run_end = index
        while run_end < len(anchors) and anchors[run_end] is None:
            run_end += 1
        next_start = anchors[run_end][0] if run_end < len(anchors) else end
        gap = _trim(display, previous_end, next_start)
        if gap is None:
            if spans:
                gap = spans[-1]
            elif run_end < len(anchors):
                gap = anchors[run_end]
            else:
                gap = _trim(display, start, end) or (start, min(end, start + 1))
        spans.extend([gap] * (run_end - index))
        shared += run_end - index
        index = run_end

    return spans, shared


def _trim(display: str, a: int, b: int) -> tuple[int, int] | None:
    while a < b and display[a] in _GAP_LEFT:
        a += 1
    while b > a and display[b - 1] in _GAP_RIGHT:
        b -= 1
    return (a, b) if b > a else None


def sentence_ids(chunk_texts: Sequence[str], sentences: Sequence[str]) -> list[list[int]]:
    """For every character of every chunk, the index of its sentence.

    Chunks are sentences packed with single spaces (long ones split at
    clauses), so their non-space characters are the sentences' non-space
    characters in the same order. Walking both together assigns each chunk
    character a sentence even when a sentence spans several chunks. Spaces
    take the sentence before them. If the two ever disagree, the walk
    resynchronises a short way ahead instead of drifting.
    """
    stream = [(char, index) for index, sentence in enumerate(sentences) for char in sentence if not char.isspace()]
    cursor = 0
    current = 0
    out: list[list[int]] = []

    for text in chunk_texts:
        ids: list[int] = []
        for char in text:
            if char.isspace():
                ids.append(current)
                continue
            if cursor < len(stream) and stream[cursor][0] == char:
                current = stream[cursor][1]
                cursor += 1
            else:
                for ahead in range(cursor + 1, min(cursor + 64, len(stream))):
                    if stream[ahead][0] == char:
                        cursor = ahead + 1
                        current = stream[ahead][1]
                        break
            ids.append(current)
        out.append(ids)
    return out


__all__ = [
    "fold",
    "is_word",
    "locate_tokens",
    "naive_display_spans",
    "sentence_ids",
    "word_core",
]
