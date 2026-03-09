import asyncio
import click
import typer
from datetime import datetime, timedelta
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from google.cloud.sql.connector import Connector

from radar.core.ingest import (
    TextIngestAgent,
    IntelligenceAgent,
    DeepResearchAgent,
    ADSBScanner,
    APRSStreamer,
    SectorScanner,
    TacticalAgent,
)
from radar.config import settings
from radar.db.engine import set_global_connector, async_session
from radar.db.init import init_db
from radar.db.models import (
    Entity,
    Connection,
    Signal,
    Trend,
    ChatSession,
    ChatMessage,
)
from sqlalchemy import select

app = typer.Typer(name="radar", help="📡 Personal Industry Intelligence Brain")
console = Console()


async def do_ask_logic(
    question: str,
    interactive: bool = False,
    session_id: Optional[str] = None,
    json_out: bool = False,
):
    import uuid
    import json

    intel = IntelligenceAgent()
    connector = None
    if settings.INSTANCE_CONNECTION_NAME:
        loop = asyncio.get_running_loop()
        connector = Connector(loop=loop)
        set_global_connector(connector)
        await connector.__aenter__()

    try:
        active_session_id = None
        if session_id:
            active_session_id = uuid.UUID(session_id)

        if interactive and not active_session_id:
            async with async_session() as session:
                new_session = ChatSession(title=question[:50] if question else "Chat")
                session.add(new_session)
                await session.commit()
                await session.refresh(new_session)
                active_session_id = new_session.id
                console.print(
                    f"[bold green]Session Started:[/bold green] {active_session_id}"
                )

        current_question = question
        while True:
            if not current_question and interactive:
                current_question = click.prompt(
                    click.style("\nRADAR> ", fg="cyan", bold=True), prompt_suffix=""
                )
                if current_question.lower() in ["exit", "quit", "bye"]:
                    break

            if not current_question:
                break

            question_vector = await intel.get_embedding(current_question)
            async with async_session() as session:
                stmt = (
                    select(Signal)
                    .order_by(Signal.vector.l2_distance(question_vector))  # type: ignore
                    .limit(5)
                )
                relevant_signals_seq = (await session.execute(stmt)).scalars().all()
                relevant_signals = list(relevant_signals_seq)
                history = []
                if active_session_id:
                    hist_stmt = (
                        select(ChatMessage)
                        .where(ChatMessage.session_id == active_session_id)  # type: ignore
                        .order_by(ChatMessage.created_at)  # type: ignore
                    )
                    for msg in (await session.execute(hist_stmt)).scalars().all():
                        history.append({"role": msg.role, "content": msg.content})

            if not relevant_signals:
                console.print("[yellow]No signals found.[/yellow]")
                if not interactive:
                    break
                current_question = ""
                continue

            if json_out:
                answer = await (
                    intel.chat(current_question, relevant_signals, history)
                    if active_session_id
                    else intel.answer_question(current_question, relevant_signals)
                )
            else:
                with console.status("[bold blue]Thinking...[/bold blue]"):
                    answer = await (
                        intel.chat(current_question, relevant_signals, history)
                        if active_session_id
                        else intel.answer_question(current_question, relevant_signals)
                    )

            if active_session_id:
                async with async_session() as session:
                    session.add(
                        ChatMessage(
                            session_id=active_session_id,
                            role="user",
                            content=current_question,
                        )
                    )
                    session.add(
                        ChatMessage(
                            session_id=active_session_id,
                            role="assistant",
                            content=answer,
                        )
                    )
                    await session.commit()

            if json_out:
                print(
                    json.dumps(
                        {
                            "question": current_question,
                            "answer": answer,
                            "sources": [{"title": s.title} for s in relevant_signals],
                        },
                        indent=2,
                    )
                )
            else:
                console.print(
                    Panel(answer, title=f"[bold cyan]{current_question}[/bold cyan]")
                )
                console.print(
                    f"[dim]Sources: {', '.join(set([s.title for s in relevant_signals]))}[/dim]"
                )

            if not interactive:
                break
            current_question = ""
    finally:
        if connector:
            await connector.__aexit__(None, None, None)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Ask a question."),
    json_out: bool = typer.Option(False, "--json", help="JSON output."),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Interactive mode."
    ),
    session_id: Optional[str] = typer.Option(None, "--session", help="Session ID."),
):
    """Ask a question or start an interactive chat."""
    asyncio.run(
        do_ask_logic(
            question=question,
            interactive=interactive,
            session_id=session_id,
            json_out=json_out,
        )
    )


@app.callback()
def main_callback():
    """📡 RADAR: Industry Intelligence & Tactical SIGINT."""
    pass


@app.command()
def sync(
    daily: bool = typer.Option(False, "--daily", "-d", help="Full research sweep."),
    tactical: bool = typer.Option(
        False, "--tactical", "-t", help="Sensor SITREP ingest."
    ),
    voice: bool = typer.Option(False, "--voice", "-v", help="Enable voice."),
):
    """Unified intelligence sync."""

    async def do_sync():
        if daily:
            import os

            targets = "sweep_targets.txt"
            if os.path.exists(targets):
                with open(targets, "r") as f:
                    topics = [line.strip() for line in f if line.strip()]
                agent = DeepResearchAgent()
                for topic in topics:
                    console.print(f"[cyan]Researching:[/cyan] {topic}")
                    try:
                        text = await agent.research(topic)
                        await run_ingest(
                            f"Title: Deep Research - {topic}\n\n{text}", voice
                        )
                    except Exception as e:
                        console.print(f"[red]Error {topic}:[/red] {e}")
        if tactical:
            console.print("[blue]Ingesting SITREP...[/blue]")
            sitrep_text = await TacticalAgent().generate_full_sitrep()
            await run_ingest(sitrep_text, voice)

    asyncio.run(do_sync())


async def run_ingest(text: str, voice: bool):
    import subprocess

    agent = TextIngestAgent()

    async def _ingest():
        connector = None
        if settings.INSTANCE_CONNECTION_NAME:
            loop = asyncio.get_running_loop()
            connector = Connector(loop=loop)
            set_global_connector(connector)
            await connector.__aenter__()
        try:
            signal, kg = await agent.ingest(text)
            items = [f"{e.name}: {e.description}" for e in kg.entities] + [
                f"{t.name}: {t.description}" for t in kg.trends
            ]
            vectors = await agent.intel.get_batch_embeddings(items) if items else []
            ent_vecs, trend_vecs = (
                vectors[: len(kg.entities)],
                vectors[len(kg.entities) :],
            )
            async with async_session() as session:
                session.add(signal)
                await session.flush()
                entity_map = {}
                for idx, e in enumerate(kg.entities):
                    exist = (
                        await session.execute(
                            select(Entity).where(Entity.name == e.name)
                        )
                    ).scalar_one_or_none()
                    if exist:
                        entity_map[e.name] = exist.id
                    else:
                        new_e = Entity(
                            name=e.name,
                            type=e.type,
                            details={"description": e.description},
                            vector=ent_vecs[idx] if idx < len(ent_vecs) else None,
                        )
                        session.add(new_e)
                        await session.flush()
                        entity_map[e.name] = new_e.id
                for idx, t in enumerate(kg.trends):
                    if not (
                        await session.execute(select(Trend).where(Trend.name == t.name))
                    ).scalar_one_or_none():
                        session.add(
                            Trend(
                                name=t.name,
                                description=t.description,
                                velocity=t.velocity,
                                vector=trend_vecs[idx]
                                if idx < len(trend_vecs)
                                else None,
                            )
                        )
                for c in kg.connections:
                    sid, tid = (
                        entity_map.get(c.source_entity_name),
                        entity_map.get(c.target_entity_name),
                    )
                    if sid and tid:
                        session.add(
                            Connection(
                                source_uuid=sid,
                                target_uuid=tid,
                                type=c.type,
                                meta_data={
                                    "description": c.description,
                                    "signal_id": str(signal.id),
                                },
                            )
                        )
                await session.commit()
            console.print(f"[green]Ingested:[/green] {signal.title}")
            if voice:
                subprocess.run(
                    [
                        "/home/chuck/bin/python3",
                        "/home/chuck/Scripts/generate_voice.py",
                        "--temp",
                        f"Signal ingested: {signal.title}",
                    ]
                )
        finally:
            if connector:
                await connector.__aexit__(None, None, None)

    await _ingest()


@app.command(hidden=True)
def ingest(
    source: Optional[str] = typer.Argument(
        None, help="Source to ingest: a file path, URL, or '-' for stdin"
    ),
    file: Optional[typer.FileText] = typer.Option(
        None, "--file", "-f", help="Legacy file option"
    ),
    voice: bool = typer.Option(
        False, "--voice", "-v", help="Enable voice confirmation."
    ),
):
    """Ingest massive textual intelligence via stdin, file, or URL."""
    import sys

    if source and (source.startswith("http://") or source.startswith("https://")):
        # Not implementing full web ingest in this block for brevity, fallback to manual text
        pass

    text = ""
    if source == "-":
        text = click.get_text_stream("stdin").read()
    elif source:
        try:
            with open(source, "r") as f:
                text = f.read()
        except Exception:
            raise typer.Exit(code=1)
    elif file:
        text = file.read()
    elif not sys.stdin.isatty():
        text = click.get_text_stream("stdin").read()
    else:
        raise typer.Exit(code=1)

    if not text.strip():
        console.print("[red]Error: Empty input.[/red]")
        raise typer.Exit(code=1)

    asyncio.run(run_ingest(text, voice))


@app.command()
def live(refresh: int = 2):
    """Tactical HUD."""

    async def do_live():
        from rich.live import Live
        from rich.table import Table
        from rich.layout import Layout
        from rich import box

        adsb, aprs, sector = ADSBScanner(), APRSStreamer(), SectorScanner()
        asyncio.create_task(aprs.start_stream())
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(Layout(name="left"), Layout(name="right"))
        layout["left"].split_column(Layout(name="adsb"), Layout(name="sector"))
        layout["right"].split_column(Layout(name="aprs"), Layout(name="news"))
        with Live(layout, screen=True, refresh_per_second=1):
            while True:
                ac_data = await adsb.get_live_data()
                metar = await sector.get_metar()
                atmos = await sector.get_atmos_weather()
                layout["header"].update(
                    Panel(
                        f"[bold yellow]📡 RADAR HUD[/bold yellow] | {datetime.now().strftime('%H:%M:%S')}",
                        style="white on blue",
                    )
                )

                ac_table = Table(box=box.SIMPLE_HEAD, expand=True)
                ac_table.add_column("Flight")
                ac_table.add_column("Alt")
                for ac in ac_data.get("aircraft", [])[:8]:
                    ac_table.add_row(
                        ac.get("flight", "Unk").strip(), f"{ac.get('alt_baro', 0)}ft"
                    )
                layout["adsb"].update(Panel(ac_table, title="ADS-B"))

                layout["sector"].update(
                    Panel(
                        f"[yellow]Atmos:[/yellow]\n{atmos}\n\n[cyan]KELM:[/cyan]\n{metar}",
                        title="Sector",
                    )
                )

                aprs_table = Table(show_header=False, box=None, expand=True)
                for p in aprs.packets[-8:]:
                    aprs_table.add_row(p[:50])
                layout["aprs"].update(Panel(aprs_table, title="APRS"))

                layout["news"].update(Panel("Scanning Headlines...", title="News"))
                await asyncio.sleep(refresh)

    try:
        asyncio.run(do_live())
    except KeyboardInterrupt:
        pass


@app.command()
def brief(voice: bool = True):
    """Strategic brief."""

    async def do_brief():
        intel = IntelligenceAgent()
        async with async_session() as session:
            last_24h = datetime.now() - timedelta(hours=24)
            signals = (
                (
                    await session.execute(
                        select(Signal).where(Signal.date >= last_24h).limit(20)
                    )
                )
                .scalars()
                .all()
            )
            entities = (await session.execute(select(Entity).limit(10))).scalars().all()
            context = {
                "signals": [s.title for s in signals],
                "entities": [e.name for e in entities],
                "trends": [],
            }
            text = await intel.generate_briefing(context)
            console.print(Panel(text, title="Briefing"))
            if voice:
                import subprocess

                subprocess.run(
                    [
                        "/home/chuck/bin/python3",
                        "/home/chuck/Scripts/generate_voice.py",
                        "--temp",
                        text,
                    ]
                )

    asyncio.run(do_brief())


@app.command()
def init():
    """Init DB."""

    async def _init():
        if settings.INSTANCE_CONNECTION_NAME:
            loop = asyncio.get_running_loop()
            conn = Connector(loop=loop)
            set_global_connector(conn)
            async with conn:
                await init_db()
        else:
            await init_db()

    asyncio.run(_init())
    console.print("[green]RADAR READY[/green]")


if __name__ == "__main__":
    app()
