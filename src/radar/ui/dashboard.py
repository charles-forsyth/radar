from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Header,
    Footer,
    Static,
    Input,
    Log,
    Label,
    ListItem,
    ListView,
)
from radar.db.engine import async_session
from radar.db.models import Signal, TacticalAlert
from sqlalchemy import select, desc
import textwrap


class SignalItem(ListItem):
    def __init__(self, title: str, content: str):
        super().__init__()
        self.signal_title = title
        self.signal_content = content

    def compose(self) -> ComposeResult:
        yield Label(f"📡 {self.signal_title}", classes="signal-title")


class RadarApp(App):
    """A Textual dashboard for Radar Intelligence."""

    CSS = """
    Screen {
        background: $surface;
    }
    .panel {
        border: solid $accent;
        background: $panel;
        padding: 1;
        margin: 1;
    }
    #feed-pane {
        width: 3fr;
    }
    #alert-pane {
        width: 1fr;
        color: $error;
    }
    #detail-pane {
        height: 2fr;
        border: solid $success;
        background: $panel;
        padding: 1;
        margin: 1;
        overflow-y: auto;
    }
    .signal-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "refresh_data", "Refresh"),
    ]

    async def on_mount(self) -> None:
        self.title = "Radar Mission Control"
        await self.action_refresh_data()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="feed-pane", classes="panel"):
                yield Label("🌐 Intelligence Feed", classes="panel-header")
                yield ListView(id="signal-list")
            with Vertical(id="alert-pane", classes="panel"):
                yield Label("⚠️ Tactical Alerts", classes="panel-header")
                yield Log(id="alert-log")

        with Vertical(id="detail-pane"):
            yield Label("📄 Document Viewer", classes="panel-header")
            yield Static(id="document-content")

        yield Input(placeholder="Search the local corpus (BM25s)...", id="search-input")
        yield Footer()

    async def action_refresh_data(self) -> None:
        """Fetch latest signals and alerts from the database."""
        try:
            async with async_session() as session:
                # Get Signals
                stmt = select(Signal).order_by(desc(Signal.date)).limit(50)  # type: ignore
                results = await session.execute(stmt)
                signals = results.scalars().all()

                list_view = self.query_one("#signal-list", ListView)
                await list_view.clear()
                for s in signals:
                    await list_view.append(SignalItem(s.title, s.content))

                # Get Alerts
                astmt = (
                    select(TacticalAlert)
                    .order_by(desc(TacticalAlert.created_at)) # type: ignore
                    .limit(20)
                )
                aresults = await session.execute(astmt)
                alerts = aresults.scalars().all()

                alert_log = self.query_one("#alert-log", Log)
                alert_log.clear()
                for a in reversed(alerts):
                    prefix = "🔴 CRITICAL" if a.severity == "CRITICAL" else "🟡 WARNING"
                    alert_log.write_line(f"{prefix} [{a.domain}]: {a.message}")
        except Exception as e:
            self.query_one("#alert-log", Log).write_line(f"DB Error: {e}")

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        """When a signal is selected, show its content in the viewer."""
        item = event.item
        if isinstance(item, SignalItem):
            viewer = self.query_one("#document-content", Static)
            wrapped = "\n".join(textwrap.wrap(item.signal_content, width=120))
            viewer.update(wrapped)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle BM25s searches."""
        query = event.value
        viewer = self.query_one("#document-content", Static)
        if not query.strip():
            return

        viewer.update(
            f"Searching BM25 index for: {query}\n[Implement BM25s hook here...]"
        )


if __name__ == "__main__":
    app = RadarApp()
    app.run()
