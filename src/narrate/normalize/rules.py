"""Turn markdown prose into text a TTS engine reads correctly.

Ordering is load-bearing and the passes are not commutative. The sequence is:

    markdown syntax -> lexicon drops -> structural noise -> currency ->
    numeric forms -> abbreviations -> acronyms -> lexicon pronunciations ->
    whitespace

Currency runs before general numbers because ``$4.2M`` must be consumed as one
unit - once ``4.2`` becomes "four point two" the dollar sign is stranded.
Acronyms run late because earlier passes introduce capitalised words that would
otherwise be mistaken for acronyms.
"""

from __future__ import annotations

import re
from functools import lru_cache

from narrate.normalize.lexicon import Lexicon

# --- shared helpers -------------------------------------------------------


@lru_cache(maxsize=1)
def _engine():
    import inflect

    return inflect.engine()


def _cardinal(value: int | float) -> str:
    return _engine().number_to_words(value, andword="", zero="zero").replace(",", "")


def _ordinal(value: int) -> str:
    return _engine().number_to_words(_engine().ordinal(value), andword="")


def _spell(word: str) -> str:
    """``FBI`` -> ``F. B. I.`` - the periods force per-letter prosody."""
    return " ".join(f"{c}." for c in word.upper())


def _say_year(value: int) -> str:
    """Read years the way people do.

    1997 is "nineteen ninety-seven", not "one thousand nine hundred and
    ninety-seven". 2007 is "two thousand seven" - the pairwise reading only
    applies when the second pair is non-zero.
    """
    if not 1100 <= value <= 2099:
        return _cardinal(value)

    high, low = divmod(value, 100)
    if low == 0:
        return f"{_cardinal(high)} hundred"
    if low < 10:
        # 2007 -> "two thousand seven"; 1907 -> "nineteen oh seven"
        if high >= 20:
            return _cardinal(value)
        return f"{_cardinal(high)} oh {_cardinal(low)}"
    return f"{_cardinal(high)} {_cardinal(low)}"


_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _roman_to_int(roman: str) -> int | None:
    total = 0
    previous = 0
    for char in reversed(roman.upper()):
        value = _ROMAN_VALUES.get(char)
        if value is None:
            return None
        total += -value if value < previous else value
        previous = max(previous, value)
    return total or None


# --- pass 1: markdown syntax ----------------------------------------------

_FENCED_CODE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*$", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^\s*>\s?", re.MULTILINE)
_LIST_BULLET = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_LIST_NUMBER = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_HRULE = re.compile(r"^\s*(?:[-*_]\s*){3,}$", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]+)`")


def strip_markdown(text: str) -> str:
    """Reduce markdown to plain prose.

    Headings keep their text and gain a full stop so the engine gives them
    closing intonation instead of running them into the next sentence.
    """
    text = _FENCED_CODE.sub(" ", text)
    text = _IMAGE.sub(" ", text)
    text = _LINK.sub(r"\1", text)
    text = _HRULE.sub("\n\n", text)
    text = _HEADING.sub(lambda m: _as_sentence(m.group(1)), text)
    text = _BLOCKQUOTE.sub("", text)
    text = _LIST_BULLET.sub("", text)
    text = _LIST_NUMBER.sub("", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = _INLINE_CODE.sub(r"\1", text)
    return text


def _as_sentence(heading: str) -> str:
    heading = heading.strip()
    if heading and heading[-1] not in ".!?:;":
        heading += "."
    return heading


# --- pass 2: structural noise ---------------------------------------------

_NUMERIC_CITATION = re.compile(r"\[\s*\d+(?:\s*[,–-]\s*\d+)*\s*\]")
_AUTHOR_CITATION = re.compile(r"\(\s*(?:see\s+)?[A-Z][A-Za-z'’-]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][A-Za-z'’-]+))?,?\s+\d{4}[a-z]?\s*\)")
_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_FOOTNOTE_MARKER = re.compile(r"(?<=[a-z.,;:])\[\^?\d{1,3}\]")


def strip_reference_noise(text: str, *, keep_citations: bool = False) -> str:
    """Remove citation markers, URLs and emails.

    Read aloud these are pure interruption: "the flock declined open bracket
    twelve close bracket". Kept behind a flag for academic use.
    """
    text = _EMAIL.sub(" ", text)
    text = _URL.sub(" ", text)
    if not keep_citations:
        text = _FOOTNOTE_MARKER.sub("", text)
        text = _NUMERIC_CITATION.sub("", text)
        text = _AUTHOR_CITATION.sub("", text)
    return text


# --- pass 3: currency -----------------------------------------------------

_MAGNITUDES = {
    "k": "thousand",
    "m": "million",
    "bn": "billion",
    "b": "billion",
    "tn": "trillion",
    "t": "trillion",
}

_CURRENCY_SYMBOLS = {"$": ("dollar", "dollars"), "£": ("pound", "pounds"), "€": ("euro", "euros"), "¥": ("yen", "yen")}

# The magnitude suffix and any space before it are grouped together, so that a
# plain "$1 and" does not have its trailing space swallowed by \s* and come out
# as "one dollarand".
_CURRENCY = re.compile(
    r"([$£€¥])\s?(\d[\d,]*(?:\.\d+)?)(?:\s*(bn|tn|[kmbt])\b)?",
    re.IGNORECASE,
)


def _expand_currency(match: re.Match[str]) -> str:
    symbol, digits, magnitude = match.group(1), match.group(2), match.group(3)
    singular, plural = _CURRENCY_SYMBOLS[symbol]
    amount = float(digits.replace(",", ""))

    if magnitude:
        spoken = _decimal_words(amount)
        return f"{spoken} {_MAGNITUDES[magnitude.lower()]} {plural}"

    # Whole units read plainly; sub-units get the "X dollars and Y cents" form
    # only for values that plausibly are money rather than a measurement.
    if amount.is_integer():
        whole = int(amount)
        return f"{_cardinal(whole)} {singular if whole == 1 else plural}"

    whole = int(amount)
    fraction = round((amount - whole) * 100)
    unit = singular if whole == 1 else plural
    if fraction:
        sub = "cent" if fraction == 1 else "cents"
        return f"{_cardinal(whole)} {unit} and {_cardinal(fraction)} {sub}"
    return f"{_cardinal(whole)} {unit}"


def _decimal_words(value: float) -> str:
    """``4.2`` -> ``four point two``. Digits after the point are read singly."""
    if float(value).is_integer():
        return _cardinal(int(value))
    whole, _, fraction = f"{value}".partition(".")
    digits = " ".join(_cardinal(int(d)) for d in fraction)
    return f"{_cardinal(int(whole))} point {digits}"


# --- pass 4: numeric forms ------------------------------------------------

_YEAR_RANGE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\s*[–—-]\s*(1[0-9]{3}|20[0-9]{2})\b")
_PERCENT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*%")
_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_DECADE = re.compile(r"\b(1[89]\d0|20[0-9]0)s\b")
_YEAR = re.compile(r"\b(1[1-9]\d{2}|20\d{2})\b")
_MAGNITUDE_NUMBER = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*(bn|tn|[kmbt])\b")
_PLAIN_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_ROMAN_HEADING = re.compile(
    r"\b((?:Chapter|Part|Book|Section|Volume|Act|Appendix)\s+)([IVXLCDM]{1,8})\b",
    re.IGNORECASE,
)


def expand_numbers(text: str) -> str:
    """Expand numeric forms, most specific pattern first."""
    text = _ROMAN_HEADING.sub(
        lambda m: f"{m.group(1)}{_roman_words(m.group(2))}", text
    )
    text = _YEAR_RANGE.sub(
        lambda m: f"{_say_year(int(m.group(1)))} to {_say_year(int(m.group(2)))}", text
    )
    text = _PERCENT.sub(
        lambda m: f"{_decimal_words(float(m.group(1).replace(',', '')))} percent", text
    )
    text = _ORDINAL.sub(lambda m: _ordinal(int(m.group(1))), text)
    text = _DECADE.sub(lambda m: _say_decade(int(m.group(1))), text)
    text = _MAGNITUDE_NUMBER.sub(
        lambda m: f"{_decimal_words(float(m.group(1).replace(',', '')))} "
        f"{_MAGNITUDES[m.group(2).lower()]}",
        text,
    )
    text = _YEAR.sub(lambda m: _say_year(int(m.group(1))), text)
    text = _PLAIN_NUMBER.sub(_expand_plain_number, text)
    return text


def _say_decade(year: int) -> str:
    """``1960s`` -> ``nineteen sixties``, not ``nineteen sixtys``.

    Only the final word is pluralised: "nineteen sixty" -> "nineteen sixties".
    """
    words = _say_year(year)
    head, _, last = words.rpartition(" ")
    plural = _engine().plural_noun(last) if last else words
    return f"{head} {plural}".strip()


def _roman_words(roman: str) -> str:
    value = _roman_to_int(roman)
    return _cardinal(value).title() if value else roman


def _expand_plain_number(match: re.Match[str]) -> str:
    raw = match.group(0).replace(",", "")
    try:
        return _decimal_words(float(raw)) if "." in raw else _cardinal(int(raw))
    except (ValueError, OverflowError):
        return match.group(0)


# --- pass 5: abbreviations ------------------------------------------------

#: Read as their expansion. Ordered longest-first at build time so "et al."
#: is not clipped by a shorter prefix.
ABBREVIATIONS: dict[str, str] = {
    "e.g.": "for example",
    "i.e.": "that is",
    "etc.": "et cetera",
    "et al.": "and others",
    "cf.": "compare",
    "vs.": "versus",
    "vs": "versus",
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miz",
    "Dr.": "Doctor",
    "Prof.": "Professor",
    "St.": "Saint",
    "Jr.": "Junior",
    "Sr.": "Senior",
    "Fig.": "Figure",
    "No.": "Number",
    "approx.": "approximately",
    "ca.": "circa",
    "p.m.": "P M",
    "a.m.": "A M",
    "&": "and",
    "%": "percent",
    "§": "section",
    "°": "degrees",
}

_ABBREV_PATTERN = re.compile(
    "|".join(re.escape(k) for k in sorted(ABBREVIATIONS, key=len, reverse=True)),
    re.IGNORECASE,
)

_LOOKUP = {k.lower(): v for k, v in ABBREVIATIONS.items()}


#: Symbols that sit flush against the preceding word ("2.5°", "R&D"). Their
#: expansions need explicit spaces or they fuse with their neighbours -
#: "two point fivedegrees". tidy() collapses any doubled spaces afterwards.
_FLUSH_SYMBOLS = frozenset({"&", "%", "§", "°"})


def expand_abbreviations(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        expansion = _LOOKUP[token.lower()]
        if token in _FLUSH_SYMBOLS:
            return f" {expansion} "
        return expansion

    return _ABBREV_PATTERN.sub(replace, text)


# --- pass 6: acronyms -----------------------------------------------------

_ACRONYM = re.compile(r"\b([A-Z]{2,6})\b")

#: Two or more consecutive all-caps words - a heading, a shouted line, or
#: front matter. ``THE END`` must not become ``T. H. E. E. N. D.``
_ALLCAPS_RUN = re.compile(r"\b[A-Z][A-Z'’]*(?:[\s-]+[A-Z][A-Z'’]*)+\b")

#: Ordinary English words that appear in caps for emphasis. Without these,
#: "He said NO" becomes "He said N. O."
COMMON_CAPS_WORDS = frozenset(
    {
        "A", "I", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "IF", "IN",
        "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US",
        "WE", "AND", "ALL", "BUT", "CAN", "FOR", "HAD", "HAS", "HER", "HIM",
        "HIS", "HOW", "ITS", "MAY", "NOT", "NOW", "ONE", "OUR", "OUT", "SHE",
        "THE", "TOO", "TWO", "WAS", "WHO", "WHY", "YES", "YET", "YOU", "END",
        "OFF", "OWN", "SAY", "SEE", "TOP", "USE", "WAY", "NEW", "OLD", "MAN",
    }
)


def _titlecase_caps_runs(text: str, lexicon: Lexicon) -> str:
    """Convert runs of consecutive all-caps words to title case.

    A run of adjacent initialisms ("FBI CIA NSA") is title-cased too, which is
    wrong - but headings and shouted dialogue outnumber those in books by a
    wide margin, and the lexicon's spell_out list overrides the result.
    """

    def replace(match: re.Match[str]) -> str:
        run = match.group(0)
        words = run.split()
        # Leave the run alone if every word is a known initialism.
        if words and all(w in lexicon.spell_out for w in words):
            return run
        return run.title()

    return _ALLCAPS_RUN.sub(replace, text)

#: Read as words without needing a lexicon entry. Every one of these would be
#: mangled into letters by the default spell-out rule.
KNOWN_WORD_ACRONYMS = frozenset(
    {
        "NASA", "NATO", "UNESCO", "UNICEF", "OPEC", "AIDS", "LASER", "RADAR",
        "SCUBA", "NAFTA", "ASCII", "JPEG", "PNG", "GIF", "SQL", "SONAR",
        "MIDI", "PIN", "SIM", "WASP", "ANZAC", "GULAG", "NIMBY",
    }
)


def expand_acronyms(text: str, lexicon: Lexicon) -> str:
    """Spell out capital runs unless they are known words.

    Defaulting to spell-out is the safer error: "N. A. S. A." is merely odd,
    while reading FBI as "fibbie" is unintelligible.
    """

    def replace(match: re.Match[str]) -> str:
        acronym = match.group(1)
        # The lexicon is checked before the built-in list in both directions:
        # a user who says NASA should be spelled must be able to override the
        # default, or the lexicon is not actually an override mechanism.
        if acronym in lexicon.spell_out:
            return _spell(acronym)
        if acronym in lexicon.say_as_word or acronym in KNOWN_WORD_ACRONYMS:
            return acronym.title()
        # An ordinary word capitalised for emphasis, not an initialism.
        if acronym in COMMON_CAPS_WORDS:
            return acronym.title()
        # A vowel-bearing 4+ letter run is usually a real word in caps
        # (a shouted word, a chapter title), not an initialism.
        if len(acronym) >= 4 and set(acronym) & set("AEIOU"):
            return acronym.title()
        return _spell(acronym)

    return _ACRONYM.sub(replace, text)


# --- pass 7: whitespace and residue ---------------------------------------

_MULTI_SPACE = re.compile(r"[ \t]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
_REPEATED_PUNCT = re.compile(r"([.,;:!?])\1+")
_ORPHAN_PUNCT = re.compile(r"^\s*[.,;:!?]+\s*$", re.MULTILINE)
_EMPTY_PARENS = re.compile(r"\(\s*\)|\[\s*\]")
_BLANK_LINES = re.compile(r"\n{3,}")


def tidy(text: str) -> str:
    """Clean up the gaps earlier passes leave behind."""
    text = _EMPTY_PARENS.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _REPEATED_PUNCT.sub(r"\1", text)
    text = _ORPHAN_PUNCT.sub("", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


# --- the pipeline ---------------------------------------------------------


class RuleNormalizer:
    """Default normalizer. Pure and deterministic - same text in, same out."""

    name = "rules"

    def __init__(
        self, lexicon: Lexicon | None = None, *, keep_citations: bool = False
    ) -> None:
        self.lexicon = lexicon or Lexicon()
        self.keep_citations = keep_citations

    def normalize(self, text: str) -> str:
        text = strip_markdown(text)
        text = strip_reference_noise(text, keep_citations=self.keep_citations)
        text = _CURRENCY.sub(_expand_currency, text)
        text = expand_numbers(text)
        text = expand_abbreviations(text)
        # Multi-word caps runs are resolved before single-word acronyms, so a
        # heading is recognised as a heading rather than a string of initials.
        text = _titlecase_caps_runs(text, self.lexicon)
        text = expand_acronyms(text, self.lexicon)
        # Lexicon pronunciations run last so they override every rule above.
        text = self.lexicon.apply_pronunciations(text)
        return tidy(text)
