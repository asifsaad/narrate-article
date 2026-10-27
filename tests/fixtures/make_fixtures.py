"""Generate test documents containing prose, a table and a figure.

Every fixture embeds three sentinel strings so tests can assert on content
rather than on formatting:

    PROSE_SENTINEL   must survive ingest
    TABLE_SENTINEL   must be gone (lives only inside table cells)
    FIGURE_SENTINEL  must be gone (lives only in the figure caption)

Run: uv run python tests/fixtures/make_fixtures.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

PROSE_SENTINEL = "quintessential"
TABLE_SENTINEL = "zbxqtable"
FIGURE_SENTINEL = "zbxqfigure"

FIXTURE_DIR = Path(__file__).parent


def make_pdf(path: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        Image,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("The Migration of Starlings", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Chapter One: Departure", styles["Heading1"]))
    story.append(
        Paragraph(
            f"The starlings gathered at dusk, a {PROSE_SENTINEL} display of "
            "collective motion that observers have recorded since 1847. Each bird "
            "tracks only its seven nearest neighbours, yet the flock as a whole "
            "banks and folds as though following a single intention. Dr. Ramirez "
            "estimated the autumn flock at $4.2M individuals, a figure later "
            "revised downward by roughly 12%. The phenomenon was documented across "
            "1939-1945 despite wartime restrictions on field research.",
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 12))

    data = [
        ["Region", "Count", "Notes"],
        ["North", "1200", TABLE_SENTINEL],
        ["South", "3400", f"{TABLE_SENTINEL} again"],
        ["West", "890", TABLE_SENTINEL],
    ]
    table = Table(data)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 18))

    # Drawn with Pillow rather than reportlab.graphics.renderPM, which needs the
    # optional rlPyCairo backend.
    from PIL import Image as PILImage, ImageDraw

    chart_path = FIXTURE_DIR / "_chart.png"
    canvas = PILImage.new("RGB", (200, 120), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([10, 20, 70, 110], fill="steelblue")
    draw.rectangle([80, 60, 140, 110], fill="indianred")
    canvas.save(chart_path)

    story.append(Image(str(chart_path), width=200, height=120))
    story.append(
        Paragraph(
            f"Figure 1: {FIGURE_SENTINEL} distribution across observation sites.",
            styles["Italic"],
        )
    )
    story.append(Spacer(1, 18))

    story.append(Paragraph("Chapter Two: Arrival", styles["Heading1"]))
    story.append(
        Paragraph(
            "By March the flocks had thinned. What remained settled into the "
            "reedbeds outside the city, e.g. the marshes north of the estuary, "
            "where they roosted through the shortest nights. The RSPB counted "
            "them each week. Prof. Adeyemi et al. argued the decline began in "
            "the 3rd quarter of 1998, though NASA satellite imagery suggests an "
            "earlier onset.",
            styles["BodyText"],
        )
    )

    SimpleDocTemplate(str(path), pagesize=LETTER).build(story)
    chart_path.unlink(missing_ok=True)


def make_epub(path: Path) -> None:
    """Hand-rolled EPUB. Avoids an extra build dependency and keeps the XHTML
    explicit so the table/figure structure under test is unambiguous."""

    container = """<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf"
    media-type="application/oebps-package+xml"/></rootfiles>
</container>"""

    opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="id">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="id">urn:uuid:test-starlings</dc:identifier>
    <dc:title>The Migration of Starlings</dc:title>
    <dc:creator>Elena Ramirez</dc:creator>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="c1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="chapter2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="c1"/>
    <itemref idref="c2"/>
  </spine>
</package>"""

    nav = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
<head><title>Contents</title></head>
<body><nav epub:type="toc"><ol>
  <li><a href="chapter1.xhtml">Chapter One: Departure</a></li>
  <li><a href="chapter2.xhtml">Chapter Two: Arrival</a></li>
</ol></nav></body></html>"""

    chapter1 = f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Chapter One</title></head>
<body>
<h1>Chapter One: Departure</h1>
<p>The starlings gathered at dusk, a {PROSE_SENTINEL} display of collective
motion that observers have recorded since 1847. Dr. Ramirez estimated the
autumn flock at $4.2M individuals, a figure later revised downward by roughly
12%. The phenomenon was documented across 1939-1945.</p>
<table>
  <tr><th>Region</th><th>Count</th><th>Notes</th></tr>
  <tr><td>North</td><td>1200</td><td>{TABLE_SENTINEL}</td></tr>
  <tr><td>South</td><td>3400</td><td>{TABLE_SENTINEL} again</td></tr>
</table>
<figure>
  <img src="chart.png" alt="{FIGURE_SENTINEL} chart"/>
  <figcaption>Figure 1: {FIGURE_SENTINEL} distribution across sites.</figcaption>
</figure>
<p>Observation continued through the winter months without interruption.</p>
</body></html>"""

    chapter2 = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>Chapter Two</title></head>
<body>
<h1>Chapter Two: Arrival</h1>
<p>By March the flocks had thinned. What remained settled into the reedbeds
outside the city, e.g. the marshes north of the estuary. Prof. Adeyemi et al.
argued the decline began in the 3rd quarter of 1998, though NASA satellite
imagery suggests an earlier onset.</p>
</body></html>"""

    with zipfile.ZipFile(path, "w") as zf:
        # mimetype must be first and stored uncompressed per the EPUB spec.
        zf.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        zf.writestr("META-INF/container.xml", container)
        zf.writestr("OEBPS/content.opf", opf)
        zf.writestr("OEBPS/nav.xhtml", nav)
        zf.writestr("OEBPS/chapter1.xhtml", chapter1)
        zf.writestr("OEBPS/chapter2.xhtml", chapter2)


if __name__ == "__main__":
    make_pdf(FIXTURE_DIR / "sample.pdf")
    make_epub(FIXTURE_DIR / "sample.epub")
    print(f"wrote sample.pdf and sample.epub to {FIXTURE_DIR}")
