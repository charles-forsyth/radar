import typer
import sys
import asyncio
import httpx
import click
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from radar.core.ingest import TextIngestAgent, WebIngestAgent
from radar.config import settings

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
        None, help="Source to ingest: a file path, URL, or '-' for stdin"
    ),
    file: Optional[typer.FileText] = typer.Option(
        None, "--file", "-f", help="Legacy file option (preferred: radar ingest [path])"
    ),
):
    """Ingest massive textual intelligence via stdin, file, or URL."""

    async def process_ingestion():
        from radar.db.engine import async_session

        # 1. Handle URL
        if source and (source.startswith("http://") or source.startswith("https://")):
            agent = WebIngestAgent()
            try:
                signal = await agent.ingest(source)
            except httpx.HTTPStatusError as e:
                console.print(
                    f"[bold red]HTTP Error {e.response.status_code}:[/bold red] {e}"
                )
                if e.response.status_code == 403:
                    console.print(
                        "[yellow]Tip: The site might be blocking bots. We're using a standard browser User-Agent now, but some strict WAFs may still block us.[/yellow]"
                    )
                raise typer.Exit(code=1)
            except httpx.RequestError as e:
                console.print(f"[bold red]Network Error:[/bold red] {e}")
                raise typer.Exit(code=1)

        else:
            # 2. Handle Text (File or Stdin)
            text = ""
            if source == "-":
                if sys.stdin.isatty():
                    console.print(
                        "[bold blue]Ready for input... (Type your text, then press Ctrl-D on a new line to finish)[/bold blue]"
                    )
                text = click.get_text_stream("stdin").read()
            elif source:
                try:
                    with open(source, "r") as f:
                        text = f.read()
                except Exception as e:
                    console.print(
                        f"[bold red]Error reading file {source}:[/bold red] {e}"
                    )
                    raise typer.Exit(code=1)
            elif file:
                text = file.read()
            elif not sys.stdin.isatty():
                text = click.get_text_stream("stdin").read()
            else:
                console.print(
                    "[bold red]Error:[/bold red] No input provided. Use 'radar ingest -', 'radar ingest [path]', or pipe text."
                )
                raise typer.Exit(code=1)

            if not text.strip():
                console.print("[bold red]Error:[/bold red] Empty input.")
                raise typer.Exit(code=1)

            agent = TextIngestAgent()
            signal = await agent.ingest(text)

        # 3. Persist to DB
        try:
            async with async_session() as session:
                session.add(signal)
                await session.commit()
                await session.refresh(signal)
        except (OSError, asyncio.TimeoutError) as e:
            console.print(
                f"[bold red]Database Connection Error:[/bold red] Could not connect to the database at {settings.DB_URL if not settings.INSTANCE_CONNECTION_NAME else settings.INSTANCE_CONNECTION_NAME}"
            )
            console.print(f"[dim]Error details: {e}[/dim]")
            raise typer.Exit(code=1)
        except Exception as e:
            console.print(f"[bold red]Database Error:[/bold red] {e}")
            raise typer.Exit(code=1)

        return signal

    signal = asyncio.run(process_ingestion())

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
    if signal.url:
        table.add_row("[bold cyan]URL[/bold cyan]", signal.url)
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
