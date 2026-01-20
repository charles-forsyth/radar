import typer
import sys
import asyncio
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from radar.core.ingest import TextIngestAgent

app = typer.Typer(name="radar", help="📡 Personal Industry Intelligence Brain")
console = Console()


@app.command()
def init():
    """Initialize the knowledge graph database."""
    console.print(
        Panel(
            "[bold green]📡 RADAR SYSTEMS INITIALIZED[/bold green]\nKnowledge Graph ready for signals."
        )
    )


@app.command()
def scan(url: str):
    """Ingest a market signal from a URL."""
    console.print(f"[bold cyan]Scanning Signal:[/bold cyan] {url}")


@app.command()
def note(text: str):
    """Log a manual strategic observation."""
    console.print(f"[bold yellow]Strategic Note Recorded:[/bold yellow] {text}")


@app.command()
def ingest(
    source: Optional[str] = typer.Argument(
        None, help="Source to ingest: a file path or '-' for stdin"
    ),
    file: Optional[typer.FileText] = typer.Option(
        None, "--file", "-f", help="Legacy file option (preferred: radar ingest [path])"
    ),
):
    """Ingest massive textual intelligence via stdin or file."""
    text = ""

    # 1. Check positional argument
    if source == "-":
        if sys.stdin.isatty():
            console.print(
                "[bold blue]Ready for input... (Press Ctrl-D when finished)[/bold blue]"
            )
        text = sys.stdin.read()
    elif source:
        try:
            with open(source, "r") as f:
                text = f.read()
        except Exception as e:
            console.print(f"[bold red]Error reading file {source}:[/bold red] {e}")
            raise typer.Exit(code=1)

    # 2. Check legacy --file option if positional wasn't used
    elif file:
        text = file.read()

    # 3. Check for piped input (automatic)
    elif not sys.stdin.isatty():
        text = sys.stdin.read()

    else:
        console.print(
            "[bold red]Error:[/bold red] No input provided. Use 'radar ingest -', 'radar ingest [path]', or pipe text."
        )
        raise typer.Exit(code=1)

    if not text.strip():
        console.print("[bold red]Error:[/bold red] Empty input.")
        raise typer.Exit(code=1)

    agent = TextIngestAgent()

    # Run async ingest and save
    async def process_signal():
        from radar.db.engine import async_session

        signal = await agent.ingest(text)

        # Persist to DB
        async with async_session() as session:
            session.add(signal)
            await session.commit()
            await session.refresh(signal)
        return signal

    signal = asyncio.run(process_signal())

    # Improved Feedback UI
    from rich.table import Table
    from rich import box

    table = Table(
        box=box.ROUNDED,
        show_header=False,
        title="[bold green]Signal Ingested Successfully[/bold green]",
    )
    table.add_row("[bold cyan]ID[/bold cyan]", str(signal.id))
    table.add_row("[bold cyan]Title[/bold cyan]", signal.title)
    table.add_row(
        "[bold cyan]Date[/bold cyan]", signal.date.strftime("%Y-%m-%d %H:%M:%S")
    )
    table.add_row("[bold cyan]Source[/bold cyan]", signal.source)
    table.add_row("[bold cyan]Length[/bold cyan]", f"{len(signal.content)} chars")

    # Add a preview snippet
    snippet = (
        signal.content[:200].replace("\n", " ") + "..."
        if len(signal.content) > 200
        else signal.content
    )
    table.add_row("[bold cyan]Preview[/bold cyan]", f"[dim]{snippet}[/dim]")

    console.print(table)
