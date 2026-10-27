"""Hand-written markdown: notes, posts, READMEs.

Each case is furniture that a renderer hides and a narrator would otherwise
read aloud. Sentinels that must not survive are nonsense words, so a pass
cannot appear to work by accident.
"""

from __future__ import annotations

import pytest

from narrate.ingest import ingest
from narrate.ingest.markdown import clean_markdown, split_front_matter
from narrate.normalize import RuleNormalizer

GONE = "zbxqgone"


def _book(tmp_path, text: str, name: str = "note.md"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return ingest(path)


def _full_text(book) -> str:
    return "\n".join(chapter.markdown for chapter in book.chapters)


# --- front matter ---------------------------------------------------------


def test_yaml_front_matter_is_metadata_not_prose(tmp_path):
    book = _book(
        tmp_path,
        f"---\ntitle: Why Sleep Matters\nauthor: Jane Doe\ntags: [{GONE}]\n---\n\n"
        "# Why Sleep Matters\n\nBody text.\n",
    )
    assert book.title == "Why Sleep Matters"
    assert book.author == "Jane Doe"
    assert book.metadata["tags"] == GONE
    assert GONE not in _full_text(book)


def test_front_matter_title_is_spoken_when_the_body_has_none(tmp_path):
    """Blog posts keep the title in front matter only."""
    book = _book(tmp_path, "---\ntitle: Morning Pages\n---\nThe body starts here.\n")
    assert book.chapters[0].markdown.startswith("# Morning Pages")


def test_front_matter_title_is_not_spoken_twice(tmp_path):
    book = _book(tmp_path, "---\ntitle: Morning Pages\n---\n# Morning Pages\n\nBody.\n")
    assert _full_text(book).count("Morning Pages") == 1


def test_toml_front_matter(tmp_path):
    book = _book(tmp_path, f'+++\ntitle = "Hugo Post"\ndraft = "{GONE}"\n+++\n\nBody.\n')
    assert book.title == "Hugo Post"
    assert GONE not in _full_text(book)


def test_author_list_is_joined():
    front, _ = split_front_matter("---\nauthors: [Ann, Bo]\n---\nBody\n")
    assert front["authors"] == ["Ann", "Bo"]


def test_leading_rule_that_is_not_a_mapping_stays_prose():
    front, body = split_front_matter("---\nJust a sentence after a rule.\n---\n")
    assert front == {}
    assert "Just a sentence" in body


def test_title_falls_back_to_the_filename(tmp_path):
    book = _book(tmp_path, "No heading at all.\n", name="sleep-notes.md")
    assert book.title == "sleep notes"


# --- block furniture ------------------------------------------------------


@pytest.mark.parametrize(
    "markup",
    [
        f"<!-- {GONE} -->",
        f"%% {GONE} %%",
        f"```python\n{GONE} = 1\n```",
        f"~~~\n{GONE}\n~~~",
        f"````\n```\n{GONE}\n```\n````",
        f"$$\n{GONE}^2\n$$",
        f"<script>{GONE}()</script>",
        f"| a | b |\n|---|---|\n| {GONE} | x |",
        f"a | b\n--- | ---\n{GONE} | x",
        f"[^1]: {GONE} footnote body\n    {GONE} continued",
        f"[ref]: https://{GONE}.example.com \"Title\"",
        f"![[{GONE}.png]]",
        f"![{GONE}](chart.png)",
        f"![{GONE}][chart]",
    ],
)
def test_non_prose_blocks_are_dropped(markup):
    text = clean_markdown(f"Before.\n\n{markup}\n\nAfter.")
    assert GONE not in text
    assert "Before." in text and "After." in text


def test_unclosed_fence_runs_to_the_end():
    assert GONE not in clean_markdown(f"Prose.\n\n```\n{GONE}\n")


def test_prose_that_resembles_a_link_definition_survives():
    assert "this matters" in clean_markdown("[Note]: this matters a great deal.")


def test_html_tags_go_but_inline_code_keeps_its_brackets():
    text = clean_markdown("Use <b>bold</b> or the `<details>` tag.<br>Next line.")
    assert text == "Use bold or the `<details>` tag.\nNext line."


# --- inline furniture -----------------------------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("See [[Circadian Rhythm|circadian rhythms]].", "See circadian rhythms."),
        ("See [[Circadian Rhythm]].", "See Circadian Rhythm."),
        ("See [[Sleep#Stages]].", "See Sleep."),
        ("Read [the guide][1] first.", "Read the guide first."),
        (f"A claim.[^{GONE}] More.", "A claim. More."),
        ("Mail <me@example.com> or visit <https://example.com>.", "Mail  or visit ."),
        ("It was ~~not~~ ==very== good.", "It was not very good."),
        (r"An \*escaped\* star.", "An *escaped* star."),
        ("Ham &amp; eggs.", "Ham & eggs."),
    ],
)
def test_inline_markup(source, expected):
    assert clean_markdown(source) == expected


def test_task_list_boxes_are_not_read():
    text = clean_markdown("- [x] Bought curtains\n- [ ] Moved the charger")
    assert text == "- Bought curtains\n- Moved the charger"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("> [!NOTE]\n> Naps help.", "> Note.\n> Naps help."),
        ("> [!tip] Drink water\n> Often.", "> Drink water.\n> Often."),
        ("> [!WARNING]- Hot\n> Careful.", "> Hot.\n> Careful."),
    ],
)
def test_callouts_are_announced_not_spelled(source, expected):
    assert clean_markdown(source) == expected


# --- setext headings ------------------------------------------------------


def test_setext_headings_become_atx():
    text = clean_markdown("Title\n=====\n\nIntro.\n\nSection\n-------\n\nBody.")
    assert text == "# Title\n\nIntro.\n\n## Section\n\nBody."


def test_rule_under_a_multiline_paragraph_is_not_a_heading():
    text = clean_markdown("First line\nsecond line\n---\n\nAfter.")
    assert "#" not in text


def test_rule_under_a_list_is_not_a_heading():
    assert "#" not in clean_markdown("- item\n---\n\nAfter.")


# --- the requirement ------------------------------------------------------


NOTE = f"""---
title: Why Sleep Matters
author: Jane Doe
---

<!-- {GONE} -->

Why Sleep Matters
=================

Most adults need sleep.[^1] See [the CDC](https://www.cdc.gov/sleep).

## The basics

- [x] Light sleep
- [ ] Deep sleep

> [!NOTE]
> Naps help.

Some <b>bold</b> text and [[Circadian Rhythm|rhythms]].

| Stage | Minutes |
|-------|---------|
| {GONE} | 90 |

```
{GONE}
```

[^1]: {GONE} footnote.
[cdc]: https://www.cdc.gov/sleep
"""


def test_no_markup_reaches_the_engine(tmp_path):
    """The whole stack, ingest through normalize: only words come out."""
    book = _book(tmp_path, NOTE)
    spoken = "\n".join(RuleNormalizer().normalize(c.markdown) for c in book.chapters)

    assert GONE not in spoken
    assert not set(spoken) & set("<>[]|`#$=~*_^")
    assert "Most adults need sleep." in spoken
    assert "Light sleep.\nDeep sleep." in spoken
    assert book.author == "Jane Doe"
    assert book.chapters[0].title == "Why Sleep Matters"  # from the setext heading
