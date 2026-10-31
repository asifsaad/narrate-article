"""Command-line interface."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from narrate.audio import format_duration
from narrate.pipeline import NarrationConfig, extract, load_checkpoint, narrate

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Turn PDFs, EPUBs and text into narrated audio using open-weights TTS.",
)
console = Console()


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Docling and its model stack are extremely chatty at INFO.
    for noisy in ("docling", "RapidOCR", "urllib3", "filelock", "numba"):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def _parse_chapter_selection(spec: str | None) -> set[int] | None:
    """Parse ``1-3,7,9-11`` into a set of chapter numbers."""
    if not spec:
        return None

    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start, end = (int(x) for x in part.split("-", 1))
            except ValueError:
                raise typer.BadParameter(f"Bad chapter range: {part!r}") from None
            if start > end:
                raise typer.BadParameter(f"Reversed chapter range: {part!r}")
            selected.update(range(start, end + 1))
        else:
            try:
                selected.add(int(part))
            except ValueError:
                raise typer.BadParameter(f"Bad chapter number: {part!r}") from None
    return selected or None


@app.command("voices")
def list_voices(engine: str = typer.Option("kokoro", help="Engine to list voices for")):
    """List available voices."""
    from narrate.tts import get_engine

    table = Table(title=f"{engine} voices", header_style="bold")
    table.add_column("id", style="cyan")
    table.add_column("name")
    table.add_column("lang")
    table.add_column("gender")
    table.add_column("notes", style="dim")

    for voice in get_engine(engine).list_voices():
        table.add_row(voice.id, voice.name, voice.language, voice.gender or "", voice.notes)

    console.print(table)


@app.command("extract")
def extract_command(
    source: Path = typer.Argument(..., exists=True, help="PDF, EPUB, DOCX, HTML or text"),
    split_level: int = typer.Option(2, "--split-level", help="Heading depth for chapters"),
    captions: bool = typer.Option(False, "--captions", help="Include figure captions"),
    footnotes: bool = typer.Option(False, "--footnotes", help="Include footnotes"),
    ocr: bool = typer.Option(False, "--ocr", help="Run OCR (for scanned documents)"),
    force_empty: bool = typer.Option(False, "--force-empty", help="Continue despite no text layer"),
    work_dir: Path = typer.Option(Path("work"), "--work-dir"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Extract to markdown only, so the text can be reviewed before synthesis."""
    _configure_logging(verbose)

    config = NarrationConfig(
        split_level=split_level,
        include_captions=captions,
        include_footnotes=footnotes,
        ocr=ocr,
        force_empty=force_empty,
        work_dir=work_dir,
    )

    with console.status(f"Extracting {source.name}..."):
        book, checkpoint = _guard(lambda: extract(source, config))

    console.print(f"[green]✓[/green] {book.title} — {len(book.chapters)} chapters, {book.word_count:,} words")
    console.print(f"  Markdown: [cyan]{checkpoint}/chapters/[/cyan]")
    console.print(f"  Narrate with: [dim]narrate speak {checkpoint} --from-markdown[/dim]")


@app.command("speak")
def speak_command(
    source: Path = typer.Argument(..., exists=True, help="Document, or a checkpoint dir with --from-markdown"),
    from_markdown: bool = typer.Option(False, "--from-markdown", help="Resume from an edited checkpoint"),
    voice: str = typer.Option("af_heart", "--voice", help="Voice id (see `narrate voices`)"),
    speed: float = typer.Option(1.0, "--speed", min=0.5, max=2.0),
    engine: str = typer.Option("kokoro", "--engine"),
    audio_format: str = typer.Option(
        "mp3", "--format", help="mp3, wav, m4a, opus, or m4b for a single audiobook file"
    ),
    bitrate: str = typer.Option(
        None, "--bitrate", help="Default 128k, or 64k for m4b (AAC, mono speech)"
    ),
    chapters: str = typer.Option(None, "--chapters", help="e.g. 1-3,7"),
    split_level: int = typer.Option(2, "--split-level"),
    captions: bool = typer.Option(False, "--captions"),
    footnotes: bool = typer.Option(False, "--footnotes"),
    keep_citations: bool = typer.Option(False, "--keep-citations"),
    ocr: bool = typer.Option(False, "--ocr"),
    force_empty: bool = typer.Option(False, "--force-empty"),
    no_loudnorm: bool = typer.Option(False, "--no-loudnorm", help="Skip loudness normalization"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Re-synthesize everything"),
    no_parallel_encode: bool = typer.Option(
        False, "--no-parallel-encode", help="Encode serially (slower; for debugging ffmpeg)"
    ),
    lexicon: Path = typer.Option(None, "--lexicon", help="Pronunciation overrides YAML"),
    work_dir: Path = typer.Option(Path("work"), "--work-dir"),
    output_dir: Path = typer.Option(Path("out"), "--output-dir"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Convert a document to narrated audio, one file per chapter."""
    _configure_logging(verbose)

    # AAC at 64k is transparent for mono speech; mp3 needs more headroom.
    if bitrate is None:
        bitrate = "64k" if audio_format == "m4b" else "128k"

    config = NarrationConfig(
        engine=engine,
        voice=voice,
        speed=speed,
        audio_format=audio_format,
        bitrate=bitrate,
        normalize_loudness=not no_loudnorm,
        split_level=split_level,
        include_captions=captions,
        include_footnotes=footnotes,
        keep_citations=keep_citations,
        ocr=ocr,
        force_empty=force_empty,
        lexicon_path=lexicon,
        work_dir=work_dir,
        output_dir=output_dir,
        use_cache=not no_cache,
        parallel_encode=not no_parallel_encode,
        chapters=_parse_chapter_selection(chapters),
    )

    if from_markdown:
        book = _guard(lambda: load_checkpoint(source, split_level=split_level))
        console.print(f"[dim]Resuming from checkpoint: {source}[/dim]")
    else:
        with console.status(f"Extracting {source.name}..."):
            book, checkpoint = _guard(lambda: extract(source, config))
        console.print(f"[dim]Markdown checkpoint: {checkpoint}/chapters/[/dim]")

    selected = book.narratable_chapters()
    if config.chapters is not None:
        selected = [c for c in selected if c.index in config.chapters]

    if not selected:
        console.print("[red]No chapters to narrate.[/red]")
        raise typer.Exit(1)

    console.print(
        f"[bold]{book.title}[/bold] — {len(selected)} chapters, "
        f"{sum(c.word_count for c in selected):,} words, voice [cyan]{voice}[/cyan]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress_bar:
        task = progress_bar.add_task("Synthesizing", total=len(selected))

        def on_progress(title: str, position: int, total: int) -> None:
            progress_bar.update(task, completed=position - 1, description=f"[{position}/{total}] {title[:40]}")

        result = _guard(lambda: narrate(book, config, progress=on_progress))
        progress_bar.update(task, completed=len(selected), description="Done")

    if not result.chapters:
        console.print("[red]Nothing was rendered.[/red]")
        raise typer.Exit(1)

    console.print()
    if result.bundle:
        size_mb = result.bundle.stat().st_size / 1_048_576
        console.print(f"[green]✓[/green] [cyan]{result.bundle}[/cyan]")
        console.print(f"  {len(result.chapters)} chapters, {size_mb:.0f} MB")
    else:
        console.print(
            f"[green]✓[/green] {len(result.chapters)} files in "
            f"[cyan]{output_dir / book.slug}/[/cyan]"
        )
    console.print(f"  Audio: {format_duration(result.total_duration)}")
    console.print(f"  Took:  {format_duration(result.elapsed_seconds)} ({result.cache_summary})")
    if result.elapsed_seconds > 0:
        console.print(f"  Speed: {result.total_duration / result.elapsed_seconds:.1f}x realtime")


@app.command("article")
def article_command(
    source: Path = typer.Argument(..., exists=True, dir_okay=False, help="Markdown or text file"),
    output: Path = typer.Option(
        None, "--output", "-o", help="Audio file to write. Default: out/<name>.mp3"
    ),
    voice: str = typer.Option("af_heart", "--voice", help="Voice id (see `narrate voices`)"),
    speed: float = typer.Option(1.0, "--speed", min=0.5, max=2.0),
    engine: str = typer.Option("kokoro", "--engine"),
    audio_format: str = typer.Option(
        None, "--format", help="mp3, m4a, opus or wav. Default: from --output, else mp3"
    ),
    bitrate: str = typer.Option("128k", "--bitrate"),
    keep_citations: bool = typer.Option(False, "--keep-citations"),
    no_loudnorm: bool = typer.Option(False, "--no-loudnorm", help="Skip loudness normalization"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Re-synthesize everything"),
    lexicon: Path = typer.Option(None, "--lexicon", help="Pronunciation overrides YAML"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Estimate the length and write the script; synthesize nothing"
    ),
    work_dir: Path = typer.Option(Path("work"), "--work-dir"),
    output_dir: Path = typer.Option(Path("out"), "--output-dir"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Narrate a short markdown or text file into a single audio file."""
    from narrate.article import (
        LONG_ARTICLE_MINUTES,
        load_article,
        plan_article,
        render_article,
        resolve_output,
    )

    _configure_logging(verbose)

    output, audio_format = _guard(
        lambda: resolve_output(
            source, output=output, audio_format=audio_format, output_dir=output_dir
        )
    )
    config = NarrationConfig(
        engine=engine,
        voice=voice,
        speed=speed,
        audio_format=audio_format,
        bitrate=bitrate,
        normalize_loudness=not no_loudnorm,
        keep_citations=keep_citations,
        lexicon_path=lexicon,
        work_dir=work_dir,
        output_dir=output_dir,
        use_cache=not no_cache,
    )

    book = _guard(lambda: load_article(source))
    plan = _guard(lambda: plan_article(book, config))
    if not plan.sections:
        console.print(f"[red]Nothing to narrate in {source.name}.[/red]")
        raise typer.Exit(1)

    estimate = plan.estimated_seconds()
    console.print(
        f"[bold]{book.title}[/bold] — {len(plan.sections)} sections, "
        f"{plan.word_count:,} words, ≈{format_duration(estimate)}, voice [cyan]{voice}[/cyan]"
    )
    if estimate > LONG_ARTICLE_MINUTES * 60:
        console.print(
            f"[yellow]That is long for one file.[/yellow] For chapters you can skip "
            f"between, try [dim]narrate speak {source} --format m4b[/dim]"
        )

    if dry_run:
        table = Table(header_style="bold", show_edge=False)
        table.add_column("#", justify="right", style="dim")
        table.add_column("section")
        table.add_column("words", justify="right")
        table.add_column("≈ length", justify="right")
        for position, section in enumerate(plan.sections, start=1):
            table.add_row(
                str(position),
                section.chapter.title,
                f"{section.word_count:,}",
                format_duration(plan.estimated_seconds(section)),
            )
        console.print(table)

        script = output.with_name(f"{output.stem}.script.txt")
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(plan.script(), encoding="utf-8")
        console.print(f"  Script: [cyan]{script}[/cyan] — exactly what the engine will read")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress_bar:
        task = progress_bar.add_task("Synthesizing", total=plan.chunk_count)
        result = _guard(
            lambda: render_article(
                plan, config, output=output, on_chunk=lambda: progress_bar.advance(task)
            )
        )
        progress_bar.update(task, description="Done")

    console.print()
    console.print(f"[green]✓[/green] [cyan]{result.path}[/cyan]")
    console.print(f"  Audio: {format_duration(result.duration_seconds)}")
    console.print(f"  Took:  {format_duration(result.elapsed_seconds)} ({result.cache_summary})")
    if result.elapsed_seconds > 0:
        console.print(f"  Speed: {result.duration_seconds / result.elapsed_seconds:.1f}x realtime")


@app.command("read")
def read_command(
    source: Path = typer.Argument(..., exists=True, dir_okay=False, help="Markdown or text file"),
    voice: str = typer.Option("af_heart", "--voice", help="Voice id (see `narrate voices`)"),
    speed: float = typer.Option(1.0, "--speed", min=0.5, max=2.0),
    engine: str = typer.Option("kokoro", "--engine"),
    audio_format: str = typer.Option(
        "m4a", "--format", help="m4a (default; most reliable sync in browsers) or mp3"
    ),
    bitrate: str = typer.Option("128k", "--bitrate"),
    embed_audio: bool = typer.Option(
        False, "--embed-audio", help="Put the audio inside index.html (one file to share)"
    ),
    open_page: bool = typer.Option(False, "--open", help="Open the page when done"),
    keep_citations: bool = typer.Option(False, "--keep-citations"),
    no_loudnorm: bool = typer.Option(False, "--no-loudnorm", help="Skip loudness normalization"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Re-synthesize everything"),
    lexicon: Path = typer.Option(None, "--lexicon", help="Pronunciation overrides YAML"),
    work_dir: Path = typer.Option(Path("work"), "--work-dir"),
    output_dir: Path = typer.Option(Path("out"), "--output-dir"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Narrate a file into audio plus a page that highlights each word as it is spoken."""
    from narrate.article import load_article, plan_article
    from narrate.readalong.render import readalong_paths, render_readalong

    _configure_logging(verbose)

    audio_format = audio_format.lower()
    paths = _guard(
        lambda: readalong_paths(source, output_dir=output_dir, audio_format=audio_format)
    )
    if audio_format == "mp3":
        console.print(
            "[yellow]Note:[/yellow] mp3 has an encoder delay that some browsers do not "
            "compensate for, so the highlight may run slightly early. m4a is exact."
        )

    config = NarrationConfig(
        engine=engine,
        voice=voice,
        speed=speed,
        audio_format=audio_format,
        bitrate=bitrate,
        normalize_loudness=not no_loudnorm,
        keep_citations=keep_citations,
        lexicon_path=lexicon,
        work_dir=work_dir,
        output_dir=output_dir,
        use_cache=not no_cache,
    )

    book = _guard(lambda: load_article(source))
    plan = _guard(lambda: plan_article(book, config))
    if not plan.sections:
        console.print(
            f"[red]Nothing to narrate in {source.name}.[/red] After removing front matter, "
            f"code, tables and other furniture, no speakable text is left."
        )
        raise typer.Exit(1)

    console.print(
        f"[bold]{book.title}[/bold] — {plan.word_count:,} words, "
        f"≈{format_duration(plan.estimated_seconds())}, voice [cyan]{voice}[/cyan]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress_bar:
        task = progress_bar.add_task("Synthesizing", total=plan.chunk_count)
        result = _guard(
            lambda: render_readalong(
                plan,
                config,
                paths=paths,
                embed_audio=embed_audio,
                on_chunk=lambda: progress_bar.advance(task),
            )
        )
        progress_bar.update(task, description="Done")

    for warning in result.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    console.print()
    console.print(f"[green]✓[/green] [cyan]{result.paths.page}[/cyan]")
    console.print(f"  Audio:   {result.paths.audio.name} ({format_duration(result.duration_seconds)})")
    console.print(f"  Timings: {result.paths.timings.name} ({result.stats.words:,} words)")
    console.print(f"  Took:    {format_duration(result.elapsed_seconds)} ({result.cache_summary})")
    if result.stats.shared_display_spans:
        console.print(
            f"  [dim]{result.stats.shared_display_spans} spoken words share a highlight with "
            f"their neighbours (numbers, abbreviations).[/dim]"
        )

    if open_page:
        import webbrowser

        webbrowser.open(result.paths.page.resolve().as_uri())


def _guard(action):
    """Turn expected failures into clean messages instead of tracebacks."""
    from narrate.ingest import NoTextLayerError, UnsupportedFormatError
    from narrate.tts import EngineUnavailableError

    try:
        return action()
    except (
        NoTextLayerError,
        UnsupportedFormatError,
        EngineUnavailableError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow] Cached chunks are kept; re-run to resume.")
        sys.exit(130)


if __name__ == "__main__":
    app()
