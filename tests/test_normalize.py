"""Normalization and segmentation.

These are the pure functions of the system, so they carry the heaviest test
load. Every case here is one a listener would notice.
"""

from __future__ import annotations

import pytest

from narrate.document import Chunk
from narrate.normalize import Lexicon, RuleNormalizer
from narrate.normalize.rules import _say_year, expand_acronyms
from narrate.segment import (
    HEADING_SILENCE_MS,
    PARAGRAPH_SILENCE_MS,
    SENTENCE_SILENCE_MS,
    segment_chapter,
    split_sentences,
)


@pytest.fixture
def normalizer() -> RuleNormalizer:
    return RuleNormalizer()


# --- numbers --------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("$4.2M", "four point two million dollars"),
        ("$1", "one dollar"),
        ("$3.50", "three dollars and fifty cents"),
        ("£20", "twenty pounds"),
        ("12%", "twelve percent"),
        ("2.5%", "two point five percent"),
        ("3rd", "third"),
        ("21st", "twenty-first"),
        ("1997", "nineteen ninety-seven"),
        ("2007", "two thousand seven"),
        ("2010", "twenty ten"),
        ("1900", "nineteen hundred"),
        ("1960s", "nineteen sixties"),
        ("1939-1945", "nineteen thirty-nine to nineteen forty-five"),
        ("Chapter IV", "Chapter Four"),
        ("Part XII", "Part Twelve"),
    ],
)
def test_numeric_expansion(normalizer, source, expected):
    assert normalizer.normalize(source) == expected


def test_currency_magnitude_keeps_following_space():
    """Regression: '$1 and' once produced 'one dollarand'."""
    assert normalizer_result("It cost $1 and then $2.") == (
        "It cost one dollar and then two dollars."
    )


def test_symbol_expansion_keeps_spacing():
    """Regression: '2.5°' once produced 'two point fivedegrees'."""
    assert normalizer_result("Rose 2.5° today.") == "Rose two point five degrees today."


def normalizer_result(text: str) -> str:
    return RuleNormalizer().normalize(text)


@pytest.mark.parametrize(
    ("year", "expected"),
    [(1847, "eighteen forty-seven"), (2000, "twenty hundred"), (1066, "one thousand sixty-six")],
)
def test_year_reading(year, expected):
    assert _say_year(year) == expected


# --- abbreviations --------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Dr. Ramirez", "Doctor Ramirez"),
        ("Prof. Adeyemi", "Professor Adeyemi"),
        ("e.g. this", "for example this"),
        ("i.e. that", "that is that"),
        ("et al. argued", "and others argued"),
        ("cats vs. dogs", "cats versus dogs"),
        ("Fig. 3", "Figure three"),
        ("R&D", "R and D"),
    ],
)
def test_abbreviation_expansion(normalizer, source, expected):
    assert normalizer.normalize(source) == expected


@pytest.mark.parametrize(
    "sentence",
    [
        "It was the first.",
        "At least.",
        "He had problems.",
        "They flew to America.",
        "He loved the piano.",
        "I said no.",
        "The canvas dried.",
    ],
)
def test_abbreviations_only_match_whole_words(normalizer, sentence):
    """Regression: "first." matched "st." and was read as "firSaint", losing
    the full stop - and with it the sentence boundary."""
    assert normalizer.normalize(sentence) == sentence


def test_number_abbreviation_needs_a_number(normalizer):
    assert normalizer.normalize("See No. 12.") == "See Number twelve."


# --- acronyms -------------------------------------------------------------


def test_initialisms_are_spelled_out(normalizer):
    assert normalizer.normalize("The FBI knew.") == "The F. B. I. knew."


def test_known_word_acronyms_are_not_spelled(normalizer):
    """NASA must read as a word; spelling it is the audible failure."""
    assert "N. A. S. A." not in normalizer.normalize("NASA imagery")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("THE END", "The End"),
        ("THE FULL PROJECT GUTENBERG LICENSE", "The Full Project Gutenberg License"),
        ("A MAD TEA-PARTY", "A Mad Tea-Party"),
        ("THE QUEEN OF HEARTS", "The Queen Of Hearts"),
    ],
)
def test_allcaps_headings_are_not_spelled_out(normalizer, source, expected):
    """Regression: 'THE END' became 'T. H. E. E. N. D.'

    All-caps headings and shouted lines are common in books, so treating every
    short capital run as an initialism destroyed real narration.
    """
    assert normalizer.normalize(source) == expected


def test_emphasised_word_is_not_spelled_out(normalizer):
    assert normalizer.normalize("He said NO and left.") == "He said No and left."


def test_isolated_initialisms_still_spell_out(normalizer):
    """The caps-run fix must not disarm the acronym rule itself."""
    assert normalizer.normalize("The FBI and the CIA met.") == (
        "The F. B. I. and the C. I. A. met."
    )


def test_lexicon_overrides_acronym_default():
    lexicon = Lexicon(say_as_word={"RSPB"}, spell_out={"NASA"})
    assert expand_acronyms("RSPB", lexicon) == "Rspb"
    assert expand_acronyms("NASA", lexicon) == "N. A. S. A."


def test_lexicon_pronunciation_wins_over_rules():
    lexicon = Lexicon(pronounce={"kubectl": "cube control"})
    result = RuleNormalizer(lexicon=lexicon).normalize("Run kubectl now.")
    assert result == "Run cube control now."


def test_lexicon_drop_removes_boilerplate():
    lexicon = Lexicon(drop={"All rights reserved."})
    result = RuleNormalizer(lexicon=lexicon).normalize("End. All rights reserved.")
    assert "rights" not in result


# --- reference noise ------------------------------------------------------


def test_citations_removed_by_default(normalizer):
    result = normalizer.normalize("The flock declined [12] sharply.")
    assert "twelve" not in result
    assert "declined sharply" in result


def test_author_year_citations_removed(normalizer):
    assert "Smith" not in normalizer.normalize("It fell (Smith, 2020) fast.")


def test_citations_can_be_kept():
    keeper = RuleNormalizer(keep_citations=True)
    assert "twelve" in keeper.normalize("The flock declined [12].")


def test_urls_and_emails_removed(normalizer):
    result = normalizer.normalize("See https://example.com or a@b.com today.")
    assert "example" not in result and "@" not in result
    assert result.startswith("See") and result.endswith("today.")


# --- markdown -------------------------------------------------------------


def test_markdown_syntax_is_stripped(normalizer):
    result = normalizer.normalize("**bold** and *italic* and `code`")
    assert result == "bold and italic and code"


def test_link_text_survives_url_removal(normalizer):
    assert normalizer.normalize("A [real link](http://x.com) here.") == (
        "A real link here."
    )


def test_heading_gains_terminal_punctuation(normalizer):
    assert normalizer.normalize("# Departure").endswith(".")


def test_heading_is_its_own_paragraph(normalizer):
    """Regression: the heading pattern swallowed the blank line after it, so
    every heading ran into its first paragraph and never got its pause."""
    text = normalizer.normalize("# Departure\n\nThe train left at dawn and kept going.")
    chunks = segment_chapter(text, chapter_index=1, max_chars=400)

    assert chunks[0].text == "Departure."
    assert chunks[0].trailing_silence_ms == HEADING_SILENCE_MS


def test_heading_keeps_a_trailing_hash(normalizer):
    assert normalizer.normalize("# Notes on C#") == "Notes on C#."


def test_list_stays_apart_from_the_paragraph_before_it(normalizer):
    text = normalizer.normalize("Sleep comes in cycles:\n\n- Light sleep\n- Deep sleep")
    assert text == "Sleep comes in cycles:\n\nLight sleep.\nDeep sleep."


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("- Light sleep\n- Deep sleep", "Light sleep.\nDeep sleep."),
        ("1. One\n2. Two", "One.\nTwo."),
        ("- **Bold item**", "Bold item."),
        ("- Already done.\n- Asking?", "Already done.\nAsking?"),
        ("- apples,\n- pears", "apples,\npears."),
        ("- A wrapped item\n  that continues", "A wrapped item\nthat continues."),
    ],
)
def test_list_items_are_read_as_separate_items(normalizer, source, expected):
    assert normalizer.normalize(source) == expected


def test_normalizer_is_deterministic(normalizer):
    source = "Dr. Smith spent $4.2M in 1997 (Jones, 2001)."
    assert normalizer.normalize(source) == normalizer.normalize(source)


def test_no_digits_survive_normalization(normalizer):
    """A digit reaching the engine is a normalization miss."""
    source = "In 1997, Dr. Smith spent $4.2M, i.e. 12% of the 3rd fund [4]."
    assert not any(char.isdigit() for char in normalizer.normalize(source))


# --- segmentation ---------------------------------------------------------


def test_sentences_split_on_boundaries():
    assert len(split_sentences("One thing. Two things! Three?")) == 3


def test_chunks_never_exceed_budget():
    text = " ".join(f"Sentence number {i} here." for i in range(200))
    chunks = segment_chapter(text, chapter_index=1, max_chars=200)
    assert chunks
    assert all(len(c.text) <= 200 for c in chunks)


def test_chunks_never_end_mid_sentence():
    text = "First sentence here. Second sentence here. Third sentence here."
    chunks = segment_chapter(text, chapter_index=1, max_chars=40)
    for chunk in chunks:
        assert chunk.text.rstrip()[-1] in ".!?"


def test_segmentation_preserves_all_words():
    text = "Alpha beta gamma. Delta epsilon zeta.\n\nEta theta iota."
    chunks = segment_chapter(text, chapter_index=1, max_chars=30)
    assert " ".join(c.text for c in chunks).split() == text.split()


def test_paragraph_boundary_gets_longer_silence():
    text = (
        "This is a reasonably long first paragraph with several words in it. "
        "It continues for a while.\n\n"
        "This is a second paragraph that also runs on for a good few words here."
    )
    chunks = segment_chapter(text, chapter_index=1, max_chars=80)
    gaps = {c.trailing_silence_ms for c in chunks[:-1]}
    assert PARAGRAPH_SILENCE_MS in gaps or SENTENCE_SILENCE_MS in gaps


def test_final_chunk_has_no_trailing_silence():
    chunks = segment_chapter("One. Two.\n\nThree.", chapter_index=1, max_chars=100)
    assert chunks[-1].trailing_silence_ms == 0


def test_oversized_sentence_splits_on_clauses():
    sentence = (
        "This clause is long, this other clause is also long, "
        "and this third clause runs on as well."
    )
    chunks = segment_chapter(sentence, chapter_index=1, max_chars=45)
    assert all(len(c.text) <= 45 for c in chunks)
    assert len(chunks) > 1


def test_empty_chapter_yields_no_chunks():
    assert segment_chapter("   \n\n  ", chapter_index=1, max_chars=100) == []


def test_chunk_order_is_contiguous():
    chunks = segment_chapter(
        "A one. B two.\n\nC three. D four.", chapter_index=3, max_chars=15
    )
    assert [c.order for c in chunks] == list(range(len(chunks)))
    assert all(c.chapter_index == 3 for c in chunks)


# --- cache keys -----------------------------------------------------------


def test_cache_key_ignores_position():
    """Moving a paragraph must not invalidate its rendered audio."""
    a = Chunk(text="Hello there.", chapter_index=1, order=0)
    b = Chunk(text="Hello there.", chapter_index=9, order=42)
    assert a.cache_key("kokoro", "af_heart", 1.0) == b.cache_key(
        "kokoro", "af_heart", 1.0
    )


@pytest.mark.parametrize(
    ("engine", "voice", "speed"),
    [("piper", "af_heart", 1.0), ("kokoro", "bf_emma", 1.0), ("kokoro", "af_heart", 1.1)],
)
def test_cache_key_tracks_render_settings(engine, voice, speed):
    chunk = Chunk(text="Hello there.", chapter_index=1, order=0)
    assert chunk.cache_key(engine, voice, speed) != chunk.cache_key(
        "kokoro", "af_heart", 1.0
    )
