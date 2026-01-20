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
    # Since we are in a synchronous Typer command, we need to run async code
    signal = asyncio.run(agent.ingest(text))

    console.print(
        Panel(
            f"[bold green] ingested:[/bold green] {signal.title}\n[dim]Length: {len(signal.content)} chars[/dim]"
        )
    )
