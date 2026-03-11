import asyncio
import click
import typer
from datetime import datetime, timedelta
from typing import Optional
from rich.console import Console
from rich.panel import Panel

from radar.core.ingest import (
    BrowserIngestAgent,
    IntelligenceAgent,
    DeepResearchAgent,
    ADSBScanner,
    APRSStreamer,
    SectorScanner,
    TacticalAgent,
    RSSIngestAgent,
)
from radar.db.engine import async_session
from radar.db.init import init_db
from radar.db.models import (
    Entity,
    Connection,
    Signal,
    Trend,
    ChatSession,
    ChatMessage,
    TacticalAlert,
)
from sqlalchemy import select, desc


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
                # Hybrid Search: Keyword + Vector
                keywords = [k.lower() for k in current_question.split() if len(k) > 3]

                # 1. Vector Search
                vector_stmt = (
                    select(Signal)
                    .order_by(Signal.vector.l2_distance(question_vector))  # type: ignore
                    .limit(5)
                )
                relevant_signals_seq = (
                    (await session.execute(vector_stmt)).scalars().all()
                )
                relevant_signals = list(relevant_signals_seq)

                # 2. Keyword Fallback/Augmentation
                if keywords:
                    kw_cond = [Signal.content.ilike(f"%{kw}%") for kw in keywords]  # type: ignore
                    from sqlalchemy import or_

                    kw_stmt = select(Signal).where(or_(*kw_cond)).limit(3)
                    kw_results = (await session.execute(kw_stmt)).scalars().all()
                    for s in reversed(kw_results):  # Add to beginning
                        if s.id not in [rs.id for rs in relevant_signals]:
                            relevant_signals.insert(0, s)
                        else:
                            # If already in vector search, move it to the top
                            relevant_signals.remove(
                                next(rs for rs in relevant_signals if rs.id == s.id)
                            )
                            relevant_signals.insert(0, s)

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
        pass


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
    daily: bool = typer.Option(
        False, "--daily", "-d", help="Run full topic research sweep (2x/day)."
    ),
    tactical: bool = typer.Option(
        False,
        "--tactical",
        "-t",
        help="Run a tactical sensor SITREP ingest (every 30m).",
    ),
    web: bool = typer.Option(
        False, "--web", "-w", help="Run dynamic web browser scrapes (every 4h)."
    ),
    voice: bool = typer.Option(False, "--voice", "-v", help="Enable voice."),
):
    """Unified intelligence sync."""

    async def do_sync():
        import os

        try:
            if daily:
                console.print(
                    "[bold blue]Starting Daily Intelligence Sweep...[/bold blue]"
                )
                # 1. Standard Deep Research Sweep
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

                # 2. Automated News Wire Ingestion
                console.print(
                    "\n[bold blue]Starting Global News Ingestion...[/bold blue]"
                )
                rss_agent = RSSIngestAgent()
                news_results = await rss_agent.sync_news()
                for signal, kg in news_results:
                    # Direct ingestion logic to avoid shell overhead
                    await save_ingest_to_db(signal, kg)
                    console.print(f"[green]Ingested News:[/green] {signal.title}")

                # 2.5 Roam Route Intel Ingestion
                console.print(
                    "\n[bold blue]Starting Roam Route Ingestion...[/bold blue]"
                )
                import subprocess

                places = ["Dewey", "Rylie", "Pam", "Alex", "Mom", "Ember", "Arcturus"]
                for place in places:
                    console.print(f"[cyan]Routing to:[/cyan] {place}")
                    try:
                        roam_cmd = ["roam", "route", place, "--weather", "-F", "gas"]
                        result = subprocess.run(
                            roam_cmd, capture_output=True, text=True
                        )
                        if result.returncode == 0:
                            # Remove ANSI escape codes
                            import re

                            clean_out = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
                            final_text = (
                                f"Title: Route Intel - to {place}\n\n{clean_out}"
                            )
                            await run_ingest(final_text, voice)
                        else:
                            console.print(
                                f"[red]Roam failed for {place}:[/red] {result.stderr}"
                            )
                    except Exception as e:
                        console.print(
                            f"[red]Roam execution error for {place}:[/red] {e}"
                        )

            if daily or web:
                # 3. Dynamic Web Browser Sweep
                dynamic_targets = "dynamic_targets.txt"
                if os.path.exists(dynamic_targets):
                    with open(dynamic_targets, "r") as f:
                        dyn_targets = [
                            line.strip() for line in f if line.strip() and "|" in line
                        ]

                    if dyn_targets:
                        console.print(
                            "[bold blue]Starting Dynamic Browser Sweep...[/bold blue]"
                        )
                        browser_agent = BrowserIngestAgent()
                        for target in dyn_targets:
                            parts = target.split("|", 1)
                            if len(parts) == 2:
                                url = parts[0].strip()
                                instructions = parts[1].strip()
                                console.print(f"[cyan]Dynamic Scrape:[/cyan] {url}")
                                try:
                                    text = await browser_agent.extract(
                                        url, instructions
                                    )
                                    final_text = f"Title: Dynamic Web Extraction - {url}\n\n{text}"
                                    await run_ingest(final_text, voice)
                                except Exception as e:
                                    console.print(
                                        f"[red]Dynamic Scrape failed for {url}:[/red] {e}"
                                    )

            if tactical:
                from sqlalchemy import select, desc
                from datetime import timedelta

                console.print(
                    "[bold blue]Executing Tactical SITREP Ingest...[/bold blue]"
                )

                # Fetch previous SITREP for delta analysis and baseline
                prev_sitrep = None
                baseline_sitreps = []
                async with async_session() as session:
                    # Previous for delta
                    stmt = (
                        select(Signal)
                        .where(Signal.title.contains("SITREP"))
                        .order_by(desc(Signal.date))
                        .limit(1)
                    )
                    result = await session.execute(stmt)
                    last_sig = result.scalar_one_or_none()
                    if last_sig:
                        prev_sitrep = last_sig.content

                    # Last 7 days for baseline
                    seven_days_ago = datetime.now() - timedelta(days=7)
                    baseline_stmt = (
                        select(Signal)
                        .where(Signal.title.contains("SITREP"))
                        .where(Signal.date >= seven_days_ago)
                        .order_by(desc(Signal.date))
                        .limit(50)
                    )
                    baseline_res = await session.execute(baseline_stmt)
                    baseline_sitreps = baseline_res.scalars().all()

                baseline_context = "\n\n".join(
                    [
                        f"--- SITREP {s.date} ---\n{s.content[:500]}"
                        for s in baseline_sitreps
                    ]
                )

                agent = TacticalAgent()
                sitrep_text = await agent.generate_full_sitrep(
                    previous_sitrep=prev_sitrep
                )

                # --- ANOMALY DETECTION ---
                intel = IntelligenceAgent()
                with console.status(
                    "[bold magenta]Tactical Sentinel is analyzing for anomalies...[/bold magenta]"
                ):
                    anomalies = await intel.detect_anomalies(
                        sitrep_text, baseline_context
                    )

                async with async_session() as session:
                    for anomaly in anomalies:
                        severity = anomaly["severity"]
                        domain = anomaly["domain"]
                        msg = anomaly["message"]

                        # Log to DB
                        new_alert = TacticalAlert(
                            domain=domain, severity=severity, message=msg
                        )
                        session.add(new_alert)

                        # High-Priority Alerts
                        if severity in ["WARNING", "CRITICAL"]:
                            console.print(
                                f"[bold red]TACTICAL ALERT [{domain}]:[/bold red] {msg}"
                            )
                            if voice:
                                alert_text = f"Captain, tactical anomaly detected in {domain} domain. {msg}"
                                subprocess.run(
                                    [
                                        "/home/chuck/bin/python3",
                                        "/home/chuck/Scripts/generate_voice.py",
                                        "--temp",
                                        alert_text,
                                    ]
                                )
                        else:
                            console.print(
                                f"[bold yellow]Tactical Note [{domain}]:[/bold yellow] {msg}"
                            )

                    await session.commit()
                # -------------------------

                await run_ingest(sitrep_text, voice)

        finally:
            pass

    asyncio.run(do_sync())


async def save_ingest_to_db(signal, kg):
    """Helper to persist a signal and its extraction to the database."""
    from radar.core.ingest import IntelligenceAgent

    intel = IntelligenceAgent()

    items = [f"{e.name}: {e.description}" for e in kg.entities] + [
        f"{t.name}: {t.description}" for t in kg.trends
    ]
    vectors = await intel.get_batch_embeddings(items) if items else []
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
                await session.execute(select(Entity).where(Entity.name == e.name))
            ).scalar_one_or_none()
            if exist:
                entity_map[e.name] = exist.id
                exist.last_seen = datetime.now()
                session.add(exist)
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
            exist_trend = (
                await session.execute(select(Trend).where(Trend.name == t.name))
            ).scalar_one_or_none()
            if exist_trend:
                exist_trend.last_seen = datetime.now()
                session.add(exist_trend)
            else:
                session.add(
                    Trend(
                        name=t.name,
                        description=t.description,
                        velocity=t.velocity,
                        vector=trend_vecs[idx] if idx < len(trend_vecs) else None,
                    )
                )
        for c in kg.connections:
            sid, tid = (
                entity_map.get(c.source_entity_name),
                entity_map.get(c.target_entity_name),
            )
            if sid and tid:
                exist_conn = (
                    await session.execute(
                        select(Connection)
                        .where(Connection.source_uuid == sid)
                        .where(Connection.target_uuid == tid)
                        .where(Connection.type == c.type)
                    )
                ).scalar_one_or_none()

                if exist_conn:
                    exist_conn.last_seen = datetime.now()
                    session.add(exist_conn)
                else:
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


async def run_ingest(text: str, voice: bool):
    from radar.core.ingest import TextIngestAgent
    import subprocess

    agent = TextIngestAgent()

    async def _ingest():
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
            pass

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
    instructions: Optional[str] = typer.Option(
        None,
        "--instructions",
        "-i",
        help="If source is a URL, navigate and extract data based on these instructions.",
    ),
):
    """Ingest massive textual intelligence via stdin, file, or URL."""
    import sys

    if source and (source.startswith("http://") or source.startswith("https://")):
        if instructions:
            console.print(f"[bold cyan]Fetching source at:[/bold cyan] {source}")

            async def run_dynamic():
                agent = BrowserIngestAgent()
                text = await agent.extract(source, instructions)
                final_text = f"Title: Web Extraction - {source}\n\n{text}"
                await run_ingest(final_text, voice)

            asyncio.run(run_dynamic())
            return
        else:
            # Not implementing simple static web ingest in this block for brevity, fallback to manual text
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
        from radar.core.ingest import NewsWire

        adsb, aprs, sector, news_wire = (
            ADSBScanner(),
            APRSStreamer(),
            SectorScanner(),
            NewsWire(),
        )
        asyncio.create_task(aprs.start_stream())
        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(Layout(name="left"), Layout(name="right"))
        layout["left"].split_column(Layout(name="adsb"), Layout(name="sector"))
        layout["right"].split_column(
            Layout(name="aprs", ratio=1),
            Layout(name="news", ratio=1),
            Layout(name="alerts", ratio=1),
        )
        with Live(layout, screen=True, refresh_per_second=1):
            while True:
                # Fetch data concurrently
                ac_data, metar, atmos, headlines = await asyncio.gather(
                    adsb.get_live_data(),
                    sector.get_metar(),
                    sector.get_atmos_weather(),
                    news_wire.get_headlines(),
                )

                # Fetch recent Tactical Alerts from DB
                alerts = []
                async with async_session() as session:
                    alert_stmt = (
                        select(TacticalAlert)
                        .order_by(desc(TacticalAlert.created_at))
                        .limit(5)
                    )
                    alert_res = await session.execute(alert_stmt)
                    alerts = alert_res.scalars().all()

                layout["header"].update(
                    Panel(
                        f"[bold yellow]📡 RADAR TACTICAL HUD[/bold yellow] | {datetime.now().strftime('%H:%M:%S')}",
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

                # News Table
                news_table = Table(show_header=False, box=None, expand=True)
                for h in headlines.split("\n")[:8]:
                    news_table.add_row(f"[dim]![/dim] {h.replace('- ', '')[:55]}")
                layout["news"].update(Panel(news_table, title="News Wire"))

                # Alerts Table
                alert_table = Table(show_header=False, box=None, expand=True)
                for a in alerts:
                    style = (
                        "red"
                        if a.severity == "CRITICAL"
                        else "yellow"
                        if a.severity == "WARNING"
                        else "white"
                    )
                    alert_table.add_row(f"[{style}]{a.domain}: {a.message}[/{style}]")
                layout["alerts"].update(Panel(alert_table, title="Tactical Alerts"))

                await asyncio.sleep(refresh)

    try:
        asyncio.run(do_live())
    except KeyboardInterrupt:
        pass


@app.command()
def brief(voice: bool = True):
    """Strategic brief."""

    async def do_brief():
        from radar.core.ingest import IntelligenceAgent, NewsWire

        intel = IntelligenceAgent()
        news_wire = NewsWire()

        try:
            async with async_session() as session:
                last_24h = datetime.now() - timedelta(hours=24)

                # 1. Fetch Strategic Signals
                sig_stmt = select(Signal).where(Signal.date >= last_24h).limit(20)
                signals = (await session.execute(sig_stmt)).scalars().all()

                # 2. Fetch Latest Tactical SITREPs (The last 2)
                sitrep_stmt = (
                    select(Signal)
                    .where(Signal.title.contains("SITREP"))
                    .order_by(desc(Signal.date))
                    .limit(2)
                )
                sitreps = (await session.execute(sitrep_stmt)).scalars().all()

                # 2.5 Fetch Latest Web Scrapes (Broadcastify, and Roam Routes)
                dynamic_stmt = (
                    select(Signal)
                    .where(
                        Signal.title.contains("Web Extraction")
                        | Signal.title.contains("DeFlock")
                        | Signal.title.contains("Route Intel")
                    )
                    .order_by(desc(Signal.date))
                    .limit(
                        15
                    )  # Grab enough to get Flock, Broadcastify, and Roam routes
                )
                dynamic_scrapes = (await session.execute(dynamic_stmt)).scalars().all()

                # 3. Fetch Recent Trends
                trend_stmt = select(Trend).order_by(desc(Trend.id)).limit(10)
                trends = (await session.execute(trend_stmt)).scalars().all()

                # 4. Fetch Global Headlines
                news_headlines = await news_wire.get_headlines()

                tactical_context = "\n\n".join(
                    [f"--- {s.title} ---\n{s.content[:1000]}" for s in sitreps]
                    + [f"--- {s.title} ---\n{s.content[:800]}" for s in dynamic_scrapes]
                )

                context = {
                    "signals": [
                        s.title
                        for s in signals
                        if "SITREP" not in s.title and "Web Extraction" not in s.title
                    ],
                    "trends": [t.name for t in trends],
                    "tactical": tactical_context,
                    "news": news_headlines,
                }

                with console.status(
                    "[bold yellow]IVXXa is synthesizing the briefing...[/bold yellow]"
                ):
                    text = await intel.generate_briefing(context)

                console.print(
                    Panel(
                        text,
                        title="[bold green]Strategic Intelligence Briefing[/bold green]",
                    )
                )

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
        finally:
            pass

    asyncio.run(do_brief())


@app.command()
def init():
    """Init DB."""

    async def _init():
        await init_db()

    asyncio.run(_init())
    console.print("[green]RADAR READY[/green]")


if __name__ == "__main__":
    app()
