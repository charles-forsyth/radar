from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from rich.align import Align
from rich import box
from datetime import datetime
from typing import List
from radar.db.models import Signal, Entity, Trend


def create_layout() -> Layout:
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )
    layout["main"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1),
    )
    layout["left"].split(
        Layout(name="trends", ratio=1),
        Layout(name="signals", ratio=2),
    )
    layout["right"].split(
        Layout(name="stats", ratio=1),
        Layout(name="entities", ratio=2),
    )
    return layout


def render_header() -> Panel:
    grid = Table.grid(expand=True)
    grid.add_column(justify="left")
    grid.add_column(justify="right")
    grid.add_row(
        "[b]RADAR[/b] Intelligence Command",
        datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    return Panel(grid, style="white on blue")


def render_stats(s_count: int, e_count: int, c_count: int, t_count: int) -> Panel:
    table = Table(show_header=False, box=None, expand=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="bold white")

    table.add_row("Signals", str(s_count))
    table.add_row("Entities", str(e_count))
    table.add_row("Connections", str(c_count))
    table.add_row("Trends", str(t_count))

    return Panel(
        Align.center(table, vertical="middle"),
        title="[bold]System Status[/bold]",
        border_style="green",
        box=box.ROUNDED,
    )


def render_trends(trends: List[Trend]) -> Panel:
    table = Table(expand=True, box=box.SIMPLE_HEAD, show_edge=False)
    table.add_column("Trend Name", style="bold yellow")
    table.add_column("Velocity", style="cyan")

    for t in trends:
        table.add_row(t.name, t.velocity)

    return Panel(
        table,
        title="[bold]Emerging Trends[/bold]",
        border_style="yellow",
        box=box.ROUNDED,
    )


def render_signals(signals: List[Signal]) -> Panel:
    table = Table(expand=True, box=box.SIMPLE_HEAD, show_edge=False)
    table.add_column("Date", style="dim", width=12)
    table.add_column("Title", style="bold white")
    table.add_column("Src", style="cyan", width=6)

    for s in signals:
        table.add_row(
            s.date.strftime("%m-%d %H:%M"),
            s.title[:60] + "..." if len(s.title) > 60 else s.title,
            s.source,
        )

    return Panel(
        table, title="[bold]Recent Signals[/bold]", border_style="blue", box=box.ROUNDED
    )


def render_entities(entities: List[Entity]) -> Panel:
    table = Table(expand=True, box=box.SIMPLE_HEAD, show_edge=False)
    table.add_column("Entity", style="white")
    table.add_column("Type", style="dim")

    for e in entities:
        type_color = "white"
        if e.type == "COMPANY":
            type_color = "orange1"
        elif e.type == "TECH":
            type_color = "green"
        elif e.type == "PERSON":
            type_color = "red"

        table.add_row(e.name, f"[{type_color}]{e.type}[/{type_color}]")

    return Panel(
        table,
        title="[bold]Key Entities[/bold]",
        border_style="magenta",
        box=box.ROUNDED,
    )


def render_dashboard(
    console: Console,
    signals: List[Signal],
    entities: List[Entity],
    trends: List[Trend],
    stats: dict,
):
    layout = create_layout()

    layout["header"].update(render_header())
    layout["stats"].update(
        render_stats(
            stats["signals"], stats["entities"], stats["connections"], stats["trends"]
        )
    )
    layout["trends"].update(render_trends(trends))
    layout["signals"].update(render_signals(signals))
    layout["entities"].update(render_entities(entities))
    layout["footer"].update(
        Panel(Align.center("[dim]Press Ctrl+C to exit[/dim]"), style="white on black")
    )

    console.print(layout)
