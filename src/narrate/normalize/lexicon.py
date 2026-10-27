"""User-controlled pronunciation overrides.

The escape hatch for everything the rules get wrong. Rules handle the general
case; the lexicon handles the specific word in *this* book that the rules will
never guess - a character name, a piece of jargon, an acronym that happens to
be pronounceable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_LEXICON_PATH = Path("config/lexicon.yaml")


@dataclass
class Lexicon:
    """Overrides applied before and after the general rules."""

    pronounce: dict[str, str] = field(default_factory=dict)
    """Literal replacement, case-insensitive whole word: ``kubectl`` -> ``cube control``."""

    spell_out: set[str] = field(default_factory=set)
    """Read letter by letter: ``FBI`` -> ``F. B. I.``"""

    say_as_word: set[str] = field(default_factory=set)
    """Read as a word despite being all caps: ``NASA``, ``NATO``."""

    drop: set[str] = field(default_factory=set)
    """Removed entirely. For boilerplate that survives extraction."""

    @classmethod
    def load(cls, path: Path | None = None) -> Lexicon:
        """Load from YAML. A missing file is normal, not an error."""
        if path is None:
            path = DEFAULT_LEXICON_PATH
        path = Path(path)

        if not path.is_file():
            log.debug("No lexicon at %s; using defaults", path)
            return cls()

        import yaml

        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"Could not parse lexicon {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Lexicon {path} must be a mapping at the top level")

        return cls(
            pronounce={str(k): str(v) for k, v in (data.get("pronounce") or {}).items()},
            spell_out={str(s).upper() for s in (data.get("spell_out") or [])},
            say_as_word={str(s).upper() for s in (data.get("say_as_word") or [])},
            drop={str(s) for s in (data.get("drop") or [])},
        )

    def apply_pronunciations(self, text: str) -> str:
        """Substitute ``pronounce`` entries and remove ``drop`` entries.

        Longest key first, so a multi-word entry is not pre-empted by one of its
        own words.
        """
        for term in sorted(self.drop, key=len, reverse=True):
            text = re.sub(rf"{re.escape(term)}", " ", text, flags=re.IGNORECASE)

        for term in sorted(self.pronounce, key=len, reverse=True):
            replacement = self.pronounce[term]
            # \b fails on terms with trailing punctuation ("Fig."), so only use
            # word boundaries where the edges are actually word characters.
            left = r"\b" if term[:1].isalnum() else ""
            right = r"\b" if term[-1:].isalnum() else ""
            text = re.sub(
                rf"{left}{re.escape(term)}{right}",
                replacement,
                text,
                flags=re.IGNORECASE,
            )

        return text
