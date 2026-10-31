"""Fill the player template: one self-contained HTML file.

The timings are inlined rather than fetched, because ``fetch()`` of a local
file fails on ``file://`` in Chrome. The audio is a relative link next to the
page, or - with ``--embed-audio`` - a base64 ``data:`` URI inside it.
"""

from __future__ import annotations

import base64
import html
import json
from importlib import resources
from pathlib import Path
from urllib.parse import quote

#: Above this, embedding makes the page slow to open; the audio is still
#: embedded if asked, with a warning.
EMBED_WARN_BYTES = 20 * 1024 * 1024

_MIME = {".m4a": "audio/mp4", ".mp3": "audio/mpeg"}


def template() -> str:
    return resources.files("narrate.readalong").joinpath("player.html").read_text(encoding="utf-8")


def inline_json(data: dict) -> str:
    """JSON safe inside ``<script>``: every ``<`` is escaped, so no
    ``</script>`` or ``<!--`` in the document text can end the element."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def audio_link(path: Path) -> str:
    """A relative URL for an audio file next to the page."""
    return quote(path.name)


def audio_data_uri(path: Path) -> str:
    mime = _MIME.get(path.suffix.lower(), "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def render_page(timings: dict, *, audio_src: str) -> str:
    page = template()
    # Order matters: the JSON goes in last, so text inside it that happens to
    # look like a placeholder is never substituted.
    page = page.replace("{{TITLE}}", html.escape(timings.get("title") or "Read-along"))
    page = page.replace("{{AUDIO_SRC}}", html.escape(audio_src, quote=True))
    return page.replace("{{TIMINGS}}", inline_json(timings))


__all__ = ["EMBED_WARN_BYTES", "audio_data_uri", "audio_link", "inline_json", "render_page"]
