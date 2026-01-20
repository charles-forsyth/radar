import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(name="radar", help="📡 Personal Industry Intelligence Brain")
console = Console()

@app.command()
def init():
    """Initialize the knowledge graph database."""
    console.print(Panel("[bold green]📡 RADAR SYSTEMS INITIALIZED[/bold green]\nKnowledge Graph ready for signals."))

@app.command()
def scan(url: str):
    """Ingest a market signal from a URL."""
    console.print(f"[bold cyan]Scanning Signal:[/bold cyan] {url}")

@app.command()
def note(text: str):
    """Log a manual strategic observation."""
    console.print(f"[bold yellow]Strategic Note Recorded:[/bold yellow] {text}")

if __name__ == "__main__":
    app()
