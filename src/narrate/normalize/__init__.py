"""Text normalization: prose in, speakable text out.

The ``Normalizer`` protocol is the seam for swapping in an FST-based normalizer
(``nemo-text-processing``) on Linux, where its pynini/OpenFst dependency
installs cleanly. The rule-based default is used everywhere because it is pure
Python and needs no compiled extensions.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from narrate.normalize.lexicon import Lexicon
from narrate.normalize.rules import RuleNormalizer

__all__ = ["Lexicon", "Normalizer", "RuleNormalizer", "get_normalizer"]


@runtime_checkable
class Normalizer(Protocol):
    name: str

    def normalize(self, text: str) -> str: ...


def get_normalizer(
    name: str = "rules",
    *,
    lexicon: Lexicon | None = None,
    keep_citations: bool = False,
) -> Normalizer:
    if name == "rules":
        return RuleNormalizer(lexicon=lexicon, keep_citations=keep_citations)
    raise ValueError(f"Unknown normalizer {name!r}. Available: rules")
