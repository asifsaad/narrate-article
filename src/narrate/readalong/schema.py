"""``timings.json``, version 1: the contract between the renderer and the page.

Hand-written validation, so the schema needs no dependency. The renderer runs
it before writing, and the tests run it on everything they produce: a page fed
an invalid file fails quietly in a browser, which is the worst place to debug.

Shape (times in seconds, 3 decimals; ``src`` = ``[start, end)`` character
offsets into ``display_text``)::

    {
      "version": 1, "title": str, "audio": str, "duration_s": float,
      "audio_offset_s": float, "voice": str, "speed": float,
      "timing_source": "engine" | "estimated" | "mixed",
      "display_text": str,
      "markup": [[a, b], ...],                     # syntax the page hides
      "blocks": [{"id", "kind", "src", "level"?, "number"?}, ...],
      "sentences": [{"id", "block", "src", "start", "end"}, ...],
      "words": [{"spoken", "start", "end", "src", "sentence"}, ...]
    }
"""

from __future__ import annotations

from numbers import Real

from narrate.readalong.display import BLOCK_KINDS

SCHEMA_VERSION = 1
TIMING_SOURCES = ("engine", "estimated", "mixed")

#: Rounding to 3 decimals can push a final end a hair past the duration.
_EPSILON = 0.0015


def validate_timings(data: object) -> list[str]:
    """Every way ``data`` breaks the v1 contract; empty when it is valid."""
    if not isinstance(data, dict):
        return ["top level must be an object"]

    errors: list[str] = []
    expect = {
        "version": int, "title": str, "audio": str, "duration_s": Real,
        "audio_offset_s": Real, "voice": str, "speed": Real, "timing_source": str,
        "display_text": str, "markup": list, "blocks": list, "sentences": list, "words": list,
    }
    for key, kind in expect.items():
        if key not in data:
            errors.append(f"missing {key!r}")
        elif not isinstance(data[key], kind) or isinstance(data[key], bool):
            errors.append(f"{key!r} must be {getattr(kind, '__name__', kind)}")
    if errors:
        return errors

    if data["version"] != SCHEMA_VERSION:
        errors.append(f"version must be {SCHEMA_VERSION}")
    if data["timing_source"] not in TIMING_SOURCES:
        errors.append(f"timing_source must be one of {TIMING_SOURCES}")
    if data["duration_s"] < 0:
        errors.append("duration_s must not be negative")

    length = len(data["display_text"])
    duration = data["duration_s"]

    def span(where: str, value, *, non_empty: bool = False) -> tuple[int, int] | None:
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in value)
        ):
            errors.append(f"{where}: src must be [start, end] integers")
            return None
        a, b = value
        if not 0 <= a <= b <= length:
            errors.append(f"{where}: src {value} outside display_text (length {length})")
            return None
        if non_empty and a == b:
            errors.append(f"{where}: src is empty")
        return a, b

    def times(where: str, item: dict) -> bool:
        start, end = item.get("start"), item.get("end")
        if not all(isinstance(v, Real) and not isinstance(v, bool) for v in (start, end)):
            errors.append(f"{where}: start and end must be numbers")
            return False
        if not 0 <= start <= end <= duration + _EPSILON:
            errors.append(f"{where}: [{start}, {end}] outside [0, {duration}]")
        if round(start, 3) != start or round(end, 3) != end:
            errors.append(f"{where}: times must have at most 3 decimals")
        return True

    for i, value in enumerate(data["markup"]):
        span(f"markup[{i}]", value)

    blocks = data["blocks"]
    previous_end = 0
    for i, block in enumerate(blocks):
        where = f"blocks[{i}]"
        if not isinstance(block, dict):
            errors.append(f"{where} must be an object")
            continue
        if block.get("id") != i:
            errors.append(f"{where}: id must be {i}")
        if block.get("kind") not in BLOCK_KINDS:
            errors.append(f"{where}: kind must be one of {BLOCK_KINDS}")
        if block.get("kind") == "heading" and block.get("level") not in range(1, 7):
            errors.append(f"{where}: a heading needs a level 1-6")
        if (bounds := span(where, block.get("src"), non_empty=True)) is not None:
            if bounds[0] < previous_end:
                errors.append(f"{where}: overlaps the block before it")
            previous_end = bounds[1]

    sentences = data["sentences"]
    for i, sentence in enumerate(sentences):
        where = f"sentences[{i}]"
        if not isinstance(sentence, dict):
            errors.append(f"{where} must be an object")
            continue
        if sentence.get("id") != i:
            errors.append(f"{where}: id must be {i}")
        if not isinstance(sentence.get("block"), int) or not 0 <= sentence["block"] < len(blocks):
            errors.append(f"{where}: block must index blocks")
        span(where, sentence.get("src"), non_empty=True)
        times(where, sentence)
        previous = sentences[i - 1] if i else None
        if isinstance(previous, dict) and sentence.get("start", 0) < previous.get("start", 0):
            errors.append(f"{where}: sentences must be in playback order")

    words = data["words"]
    used: set[int] = set()
    previous_start = -1.0
    for i, word in enumerate(words):
        where = f"words[{i}]"
        if not isinstance(word, dict):
            errors.append(f"{where} must be an object")
            continue
        if not isinstance(word.get("spoken"), str) or not word["spoken"]:
            errors.append(f"{where}: spoken must be a non-empty string")
        span(where, word.get("src"), non_empty=True)
        if times(where, word):
            if word["start"] < previous_start:
                errors.append(f"{where}: words must be sorted by start")
            previous_start = word["start"]
        sid = word.get("sentence")
        if not isinstance(sid, int) or not 0 <= sid < len(sentences):
            errors.append(f"{where}: sentence must index sentences")
            continue
        used.add(sid)
        sentence = sentences[sid]
        if (
            isinstance(sentence, dict)
            and isinstance(word.get("start"), Real)
            and not sentence.get("start", 0) <= word["start"] <= sentence.get("end", 0)
        ):
            errors.append(f"{where}: starts outside its sentence")

    for i in range(len(sentences)):
        if i not in used:
            errors.append(f"sentences[{i}] has no words")

    return errors


__all__ = ["SCHEMA_VERSION", "TIMING_SOURCES", "validate_timings"]
