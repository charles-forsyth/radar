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
    file: Optional[typer.FileText] = typer.Option(
        None, "--file", "-f", help="File to ingest (defaults to stdin if piped)"
    ),
):
    """Ingest massive textual intelligence via stdin or file."""
    text = ""
    if file:
        text = file.read()
    elif not sys.stdin.isatty():
        # Read from stdin if piped
        text = sys.stdin.read()
    else:
        console.print(
            "[bold red]Error:[/bold red] No input provided. Pipe text to this command or use --file."
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


if __name__ == "__main__":
    app()
