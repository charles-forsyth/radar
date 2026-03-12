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
    TacticalAgent,
    RSSIngestAgent,
    TextIngestAgent,
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
    Telemetry,
    RiverLevel,
    RFPeak,
    SoftwareInventory,
    Statistic,
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

            with console.status("[bold blue]Searching...[/bold blue]"):
                relevant_signals = await intel.search_signals(current_question, limit=5)

            async with async_session() as session:
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

        # Suppress Hugging Face verbosity
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        # Instantiate IntelligenceAgent ONCE here so the model stays warm
        from radar.core.ingest import IntelligenceAgent

        shared_intel = IntelligenceAgent()

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
                    agent = DeepResearchAgent(intel=shared_intel)

                    # Process topics concurrently with a Semaphore to prevent exploding memory
                    sem = asyncio.Semaphore(5)

                    async def process_topic(topic):
                        async with sem:
                            console.print(f"[cyan]Researching:[/cyan] {topic}")
                            try:
                                text = await agent.research(topic)
                                await run_ingest(
                                    f"Title: Deep Research - {topic}\n\n{text}",
                                    voice,
                                    shared_intel,
                                )
                            except Exception as e:
                                console.print(f"[red]Error {topic}:[/red] {e}")

                    tasks = [process_topic(t) for t in topics]
                    await asyncio.gather(*tasks)

                # 2. Automated News Wire Ingestion
                console.print(
                    "\n[bold blue]Starting Global News Ingestion...[/bold blue]"
                )
                rss_agent = RSSIngestAgent(intel=shared_intel)
                news_results = await rss_agent.sync_news()
                for signal, kg in news_results:
                    # Direct ingestion logic to avoid shell overhead
                    await save_ingest_to_db(signal, kg, shared_intel)
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
                            await run_ingest(final_text, voice, shared_intel)
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
                        browser_agent = BrowserIngestAgent(intel=shared_intel)
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
                                    await run_ingest(final_text, voice, shared_intel)
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

                # Fetch previous SITREP for baseline
                baseline_sitreps = []
                async with async_session() as session:
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
                snapshot = await agent.generate_snapshot()
                sitrep_text = snapshot.raw_sitrep

                # --- ANOMALY DETECTION ---
                with console.status(
                    "[bold magenta]Tactical Sentinel is analyzing for anomalies...[/bold magenta]"
                ):
                    anomalies = await shared_intel.detect_anomalies(
                        sitrep_text, baseline_context
                    )

                async with async_session() as session:
                    # 1. SAVE STRUCTURED TELEMETRY
                    session.add(
                        Telemetry(
                            temp_f=snapshot.temp_f,
                            aircraft_count=snapshot.aircraft_count,
                            lan_device_count=snapshot.lan_device_count,
                            ssh_failure_count=snapshot.ssh_failure_count,
                            internet_latency_ms=snapshot.internet_latency_ms,
                        )
                    )

                    for r in snapshot.rivers:
                        session.add(
                            RiverLevel(
                                station_name=r["name"], value=r["value"], unit=r["unit"]
                            )
                        )

                    for p in snapshot.rf_peaks:
                        session.add(
                            RFPeak(frequency_mhz=p["freq"], power_db=p["power"])
                        )

                    for manager, count in snapshot.software.items():
                        session.add(
                            SoftwareInventory(manager=manager, package_count=count)
                        )

                    # 2. SAVE ALERTS
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

                            import subprocess

                            try:
                                urgency = (
                                    "critical" if severity == "CRITICAL" else "normal"
                                )
                                subprocess.run(
                                    [
                                        "notify-send",
                                        "-u",
                                        urgency,
                                        f"Radar Alert: {domain}",
                                        msg,
                                    ],
                                    check=False,
                                )
                            except Exception:
                                pass

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

                await run_ingest(sitrep_text, voice, shared_intel)

        finally:
            pass

    asyncio.run(do_sync())


async def save_ingest_to_db(signal, kg, intel: IntelligenceAgent):
    """Helper to persist a signal and its extraction to the database."""
    items = [f"{e.name}: {e.description}" for e in kg.entities] + [
        f"{t.name}: {t.description}" for t in kg.trends
    ]
    # Re-embed the main signal content for semantic search
    signal.vector = await intel.get_embedding(
        f"{signal.title}\n{signal.content[:2000]}"
    )

    # NEW: Extract structured statistics from content
    extracted_stats = intel.extract_stats(signal.content)

    vectors = await intel.get_batch_embeddings(items) if items else []
    ent_vecs, trend_vecs = (
        vectors[: len(kg.entities)],
        vectors[len(kg.entities) :],
    )
    async with async_session() as session:
        session.add(signal)
        await session.flush()

        # Save extracted stats
        for s in extracted_stats:
            # Categorize based on title keywords
            category = "GENERAL"
            t_lower = signal.title.lower()
            if any(
                k in t_lower
                for k in ["price", "cost", "economic", "finance", "route intel", "gas"]
            ):
                category = "FINANCE"
            elif any(k in t_lower for k in ["benchmark", "performance", "fps", "t/s"]):
                category = "TECH_METRICS"
            elif any(k in t_lower for k in ["capacity", "count", "stats"]):
                category = "LOGISTICS"

            session.add(
                Statistic(
                    category=category,
                    label=s["label"],
                    value=s["value"],
                    unit=s["unit"],
                    source_signal_id=signal.id,
                )
            )

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
                                .where(Connection.source_uuid == sid)  # type: ignore
                                .where(Connection.target_uuid == tid)  # type: ignore
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

            try:
                await session.commit()
            except Exception as e:
                await session.rollback()
                # Ignore duplicate key errors caused by concurrent entity creation
                if "duplicate key value violates unique constraint" not in str(e):
                    raise


async def run_ingest(
    text: str, voice: bool, shared_intel: Optional[IntelligenceAgent] = None
):
    import subprocess

    agent = TextIngestAgent(intel=shared_intel)

    async def _ingest():
        try:
            signal, kg = await agent.ingest(text)
            # Use shared intel or the agent's internal one
            intel = shared_intel if shared_intel else agent.intel
            await save_ingest_to_db(signal, kg, intel)

            console.print(f"[green]Ingested:[/green] {signal.title}")

            if voice:
                alert_text = f"New intelligence report ingested: {signal.title}"
                subprocess.run(
                    [
                        "/home/chuck/bin/python3",
                        "/home/chuck/Scripts/generate_voice.py",
                        "--temp",
                        alert_text,
                    ]
                )
        except Exception as e:
            if "duplicate key value" not in str(e):
                console.print(f"[red]Ingest failed:[/red] {e}")

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
def report(
    open_browser: bool = typer.Option(
        True, "--open", help="Open the report in browser."
    ),
):
    """Generate a high-fidelity 'AI-Powered Command Center' briefing with deep data integration."""
    import jinja2
    from sqlalchemy import select, desc
    import re
    import asyncio

    async def _report():
        console.print("[bold blue]Forging v0.33 AI Executive Briefing...[/bold blue]")

        from radar.core.ingest import IntelligenceAgent

        intel = IntelligenceAgent()

        async with async_session() as session:
            # 1. Fetch Core Relational Data
            tel_stmt = select(Telemetry).order_by(desc(Telemetry.timestamp)).limit(2)
            tel_results = (await session.execute(tel_stmt)).scalars().all()
            curr_tel = tel_results[0] if len(tel_results) > 0 else None

            river_stmt = (
                select(RiverLevel).order_by(desc(RiverLevel.timestamp)).limit(60)
            )
            river_results = (await session.execute(river_stmt)).scalars().all()

            rf_stmt = select(RFPeak).order_by(desc(RFPeak.timestamp)).limit(20)
            rf_peaks_all = (await session.execute(rf_stmt)).scalars().all()

            alert_stmt = (
                select(TacticalAlert).order_by(desc(TacticalAlert.created_at)).limit(10)
            )
            alerts = (await session.execute(alert_stmt)).scalars().all()

            stats_stmt = select(Statistic).order_by(desc(Statistic.timestamp)).limit(60)
            all_stats = (await session.execute(stats_stmt)).scalars().all()

            sig_stmt = select(Signal).order_by(desc(Signal.date)).limit(50)
            recent_signals = (await session.execute(sig_stmt)).scalars().all()

        # Calculate Security Score
        ssh_count = curr_tel.ssh_failure_count if curr_tel else 0
        alert_count = len([a for a in alerts if a.severity in ["WARNING", "CRITICAL"]])
        sec_score = max(0, 100 - (ssh_count * 1.5) - (alert_count * 10))

        # Process Rivers
        station_data = {}
        for r in river_results:
            if r.station_name not in station_data:
                station_data[r.station_name] = []
            if len(station_data[r.station_name]) < 2:
                station_data[r.station_name].append(r)

        latest_rivers = []
        for name, items in station_data.items():
            curr, prev = items[0], (items[1] if len(items) > 1 else None)
            latest_rivers.append(
                {
                    "name": name,
                    "val": curr.value,
                    "unit": curr.unit,
                    "delta": curr.value - prev.value if prev else 0,
                }
            )

        # Process RF Peaks
        peak_map = {}
        for r in rf_peaks_all:
            f_key = round(r.frequency_mhz, 1)
            if f_key not in peak_map:
                peak_map[f_key] = []
            if len(peak_map[f_key]) < 2:
                peak_map[f_key].append(r)
        processed_rf = []
        for f, items in peak_map.items():
            c, p = items[0], (items[1] if len(items) > 1 else None)
            processed_rf.append(
                {
                    "freq": c.frequency_mhz,
                    "db": c.power_db,
                    "delta": c.power_db - p.power_db if p else 0,
                }
            )
        processed_rf.sort(key=lambda x: x["db"], reverse=True)

        # Process Research Stats
        stats_map = {}
        for s in all_stats:
            key = (s.category, s.label)
            if key not in stats_map:
                stats_map[key] = []
            if len(stats_map[key]) < 2:
                stats_map[key].append(s)
        categorized_stats = {}
        for key, items in stats_map.items():
            cat, label = key
            curr, prev = items[0], (items[1] if len(items) > 1 else None)
            if cat not in categorized_stats:
                categorized_stats[cat] = []
            categorized_stats[cat].append(
                {
                    "label": label,
                    "val": curr.value,
                    "unit": curr.unit or "",
                    "delta": curr.value - prev.value if prev else 0,
                }
            )

        # AI STRATEGIC ASSESSMENT
        with console.status(
            "[bold magenta]Local Intelligence Engine is performing deep sector analysis...[/bold magenta]"
        ):
            questions = [
                "Identify the top 3 tactical threats or technical vulnerabilities discovered in recent reports.",
                "Summarize the most significant regional infrastructure updates (Rail, Power, Emergency Services).",
                "Extract the 3 most important hardware or AI technology breakthroughs from the latest ingest.",
            ]

            async def ask_brain(q):
                ans = await intel.answer_question(q, recent_signals)
                return re.sub(r"🎯.*?\\n|📌.*?\\n", "", ans).strip()

            ai_results = await asyncio.gather(*[ask_brain(q) for q in questions])

        report_css = """
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=JetBrains+Mono:wght@400;700&display=swap');
        body { background-color: #030508; color: #00ff41; font-family: 'JetBrains Mono', monospace; margin: 0; padding: 20px; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }
        .dashboard { display: grid; grid-template-columns: 350px 1fr 350px; gap: 20px; }
        .header { grid-column: 1 / span 3; border: 2px solid #00ff41; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; background: #00ff4111; box-shadow: 0 0 20px #00ff4133; margin-bottom: 20px; }
        .header-title { font-family: 'Orbitron', sans-serif; font-size: 1.8rem; font-weight: 900; }
        .box { border: 1px solid #00ff41; background: #080c12; padding: 20px; position: relative; margin-bottom: 20px; box-shadow: inset 0 0 15px #00ff4111; }
        .box-label { position: absolute; top: -10px; left: 20px; background: #030508; border: 1px solid #00ff41; padding: 0 10px; color: #00ff41; font-weight: bold; font-size: 10px; }
        .big-val { font-size: 3.5rem; font-weight: 900; text-align: center; text-shadow: 0 0 25px #00ff41; margin: 10px 0; }
        .score-box { background: #00ff4111; border: 1px solid #00ff41; padding: 10px; text-align: center; font-size: 9px; }
        .ai-dispatch { background: #0a0f18; border-left: 5px solid #00ff41; padding: 20px; margin-bottom: 25px; }
        .ai-title { color: #00ff41; font-weight: bold; font-size: 14px; border-bottom: 1px solid #004111; padding-bottom: 10px; margin-bottom: 15px; }
        .ai-body { color: #cfd2d6; text-transform: none; line-height: 1.6; font-size: 13px; }
        .stat-row { display: flex; justify-content: space-between; border-bottom: 1px solid #002105; padding: 8px 0; }
        .stat-label { color: #008f11; }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
        """

        template = jinja2.Template("""
        <html>
        <head><style>{{ css }}</style><title>RADAR COMMAND CENTER</title></head>
        <body>
            <div class="dashboard">
                <div class="header">
                    <div class="header-title"><span class="pulse">●</span> RADAR COMMAND CENTER</div>
                    <div style="text-align: right; font-weight: bold;">SECTOR: TIOGA PA | {{ now }} | v{{ version }}</div>
                </div>
                <div class="col">
                    <div class="box">
                        <div class="box-label">ATMOSPHERICS</div>
                        <div class="big-val">{{ tel.temp_f }}°F</div>
                    </div>
                    <div class="box">
                        <div class="box-label">SECURITY INTEGRITY</div>
                        <div class="big-val" style="color: {% if score > 80 %}#00ff41{% elif score > 50 %}#ffff00{% else %}#ff3131{% endif %};">{{ score }}%</div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            <div class="score-box">SSH FAILS: {{ tel.ssh_failure_count }}</div>
                            <div class="score-box">ALERTS: {{ alerts|length }}</div>
                        </div>
                    </div>
                    <div class="box">
                        <div class="box-label">HYDROLOGY</div>
                        {% for r in rivers %}<div class="stat-row"><span class="stat-label">{{ r.name[:20] }}</span><span>{{ r.val }} {{ r.unit }} ({% if r.delta > 0 %}+{% endif %}{{ "%.2f"|format(r.delta) }})</span></div>{% endfor %}
                    </div>
                </div>
                <div class="col">
                    <div class="box">
                        <div class="box-label">AI EXECUTIVE DISPATCH</div>
                        <div class="ai-dispatch"><div class="ai-title">>> STRATEGIC THREATS</div><div class="ai-body">{{ ai[0] }}</div></div>
                        <div class="ai-dispatch"><div class="ai-title">>> INFRASTRUCTURE</div><div class="ai-body">{{ ai[1] }}</div></div>
                        <div class="ai-dispatch"><div class="ai-title">>> TACTICAL TECH</div><div class="ai-body">{{ ai[2] }}</div></div>
                    </div>
                    <div class="box">
                        <div class="box-label">STRATEGIC DATA TRENDS</div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                            {% for cat, items in stats.items() %}
                            <div>
                                <div style="color: #000; background: #00ff41; padding: 2px 5px; font-weight: bold; margin-bottom: 10px; font-size: 9px;">{{ cat }}</div>
                                {% for s in items %}<div class="stat-row" style="font-size: 9px;"><span class="stat-label">{{ s.label }}</span><span>{{ s.val }} {{ s.unit }}</span></div>{% endfor %}
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
                <div class="col">
                    <div class="box">
                        <div class="box-label">AIRSPACE</div>
                        <div class="big-val">{{ tel.aircraft_count }}</div>
                    </div>
                    <div class="box">
                        <div class="box-label">SIGINT SPECTRUM</div>
                        {% for f in rf[:12] %}<div class="stat-row"><span class="stat-label">{{ f.freq }} MHz</span><span>{{ f.db }} dB ({% if f.delta > 0 %}+{% endif %}{{ "%.2f"|format(f.delta) }})</span></div>{% endfor %}
                    </div>
                    <div class="box" style="border-color: #ff3131;">
                        <div class="box-label" style="color:#ff3131; border-color:#ff3131;">THREAT LOG</div>
                        {% for a in alerts[:6] %}<div style="border-bottom: 1px solid #ff313133; padding: 5px 0; color: #ff3131; font-size: 10px;">[{{ a.severity }}] {{ a.message }}</div>{% endfor %}
                    </div>
                </div>
            </div>
        </body>
        </html>
        """)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        html = template.render(
            css=report_css,
            tel=curr_tel,
            score=int(sec_score),
            rivers=latest_rivers,
            rf=processed_rf,
            stats=categorized_stats,
            ai=ai_results,
            alerts=alerts,
            now=now_str,
            version="0.33.0",
        )

        fname = "tactical_intelligence_briefing.html"
        with open(fname, "w") as f:
            f.write(html)
        console.print(f"[bold green]AI Command Briefing forged: {fname}[/bold green]")
        if open_browser:
            import subprocess

            try:
                subprocess.run(["xdg-open", fname], check=False)
            except Exception:
                pass

    asyncio.run(_report())


@app.command()
def map():
    """Generate an offline HTML map of extracted tactical coordinates."""
    import folium
    from sqlalchemy import select
    import re

    async def _map():
        console.print("[bold blue]Generating Offline Tactical Map...[/bold blue]")
        # Base map centered exactly on 1539 Button Hill Road, Tioga, PA 16946
        tioga_coords = [41.9168, -77.1042]
        m = folium.Map(location=tioga_coords, zoom_start=7, tiles="CartoDB positron")

        # Add the 150-mile strategic sector geofence (150 miles = 241,401 meters)
        folium.Circle(
            radius=241401,
            location=tioga_coords,
            popup="150-Mile Strategic Sector",
            color="#3186cc",
            fill=True,
            fill_color="#3186cc",
            fill_opacity=0.1,
        ).add_to(m)

        # Add a precise home pin
        folium.Marker(
            tioga_coords,
            popup="1539 Button Hill Road (Home Base)",
            icon=folium.Icon(color="green", icon="home"),
        ).add_to(m)

        async with async_session() as session:
            # 1. Look for ADS-B coordinates in SITREPs
            stmt = select(Signal).where(Signal.title.contains("SITREP")).limit(100)
            sitreps = (await session.execute(stmt)).scalars().all()

            for s in sitreps:
                # Naive coordinate extraction for demo purposes
                # We expect lines like: - Flight UAL450 at 38000ft (Lat: 41.5, Lon: -76.8)
                matches = re.finditer(
                    r"Lat:\s*([\-\d\.]+),\s*Lon:\s*([\-\d\.]+)", s.content
                )
                for match in matches:
                    lat, lon = float(match.group(1)), float(match.group(2))
                    folium.Marker(
                        [lat, lon],
                        popup="Aircraft Ping",
                        icon=folium.Icon(color="red", icon="plane"),
                    ).add_to(m)

            # 2. Add fixed grid infrastructure from database config
            # (Mocked for demonstration, would normally query a static infrastructure table)
            folium.Marker(
                [41.9, -77.2],
                popup="Water Treatment",
                icon=folium.Icon(color="blue", icon="tint"),
            ).add_to(m)
            folium.Marker(
                [41.7, -77.0],
                popup="Substation Alpha",
                icon=folium.Icon(color="orange", icon="bolt"),
            ).add_to(m)

        map_file = "radar_map.html"
        m.save(map_file)
        console.print(f"[bold green]Map generated at: {map_file}[/bold green]")

    asyncio.run(_map())


@app.command()
def graph():
    """Generate an offline 3D Knowledge Graph HTML file."""
    import json

    async def _graph():
        console.print("[bold blue]Compiling 3D Knowledge Graph...[/bold blue]")
        async with async_session() as session:
            # Fetch all entities
            estmt = select(Entity)
            entities = (await session.execute(estmt)).scalars().all()

            # Fetch all connections
            cstmt = select(Connection)
            connections = (await session.execute(cstmt)).scalars().all()

        nodes = []
        node_ids = set()

        for e in entities:
            color = (
                "#ff7f0e"
                if e.type.name == "COMPANY"
                else "#2ca02c"
                if e.type.name == "TECH"
                else "#d62728"
                if e.type.name == "PERSON"
                else "#9467bd"
            )
            nodes.append(
                {
                    "id": str(e.id),
                    "name": e.name,
                    "val": 1,
                    "color": color,
                    "desc": e.details.get("description", ""),
                }
            )
            node_ids.add(str(e.id))

        links = []
        for c in connections:
            if str(c.source_uuid) in node_ids and str(c.target_uuid) in node_ids:
                links.append(
                    {
                        "source": str(c.source_uuid),
                        "target": str(c.target_uuid),
                        "name": c.type.name,
                        "desc": c.meta_data.get("description", ""),
                    }
                )

        graph_data = {"nodes": nodes, "links": links}

        # Write standalone HTML file using 3d-force-graph
        html_content = f"""
        <head>
          <style> body {{ margin: 0; }} </style>
          <script src="https://unpkg.com/3d-force-graph"></script>
        </head>
        <body>
          <div id="3d-graph"></div>
          <script>
            const gData = {json.dumps(graph_data)};
            const Graph = ForceGraph3D()
              (document.getElementById('3d-graph'))
                .graphData(gData)
                .nodeLabel('name')
                .nodeAutoColorBy('group')
                .onNodeClick(node => window.alert(node.name + ": " + node.desc));
          </script>
        </body>
        """

        graph_file = "radar_graph.html"
        with open(graph_file, "w") as f:
            f.write(html_content)

        console.print(
            f"[bold green]3D Knowledge Graph generated at: {graph_file}[/bold green]"
        )

    asyncio.run(_graph())


@app.command()
def serve(port: int = typer.Option(8080, help="Port to run the sync server on.")):
    """Run a local API server for P2P intelligence mirroring."""
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    api = FastAPI(title="Radar Mesh Node")

    @api.get("/api/sync/sitrep")
    async def sync_sitrep():
        async with async_session() as session:
            stmt = (
                select(Signal)
                .where(Signal.title.contains("SITREP"))
                .order_by(desc(Signal.date))
                .limit(5)
            )
            signals = (await session.execute(stmt)).scalars().all()
            return JSONResponse(
                {
                    "sitreps": [
                        {"date": str(s.date), "title": s.title, "content": s.content}
                        for s in signals
                    ]
                }
            )

    @api.get("/api/sync/alerts")
    async def sync_alerts():
        async with async_session() as session:
            stmt = (
                select(TacticalAlert).order_by(desc(TacticalAlert.created_at)).limit(10)
            )
            alerts = (await session.execute(stmt)).scalars().all()
            return JSONResponse(
                {
                    "alerts": [
                        {
                            "date": str(a.created_at),
                            "domain": a.domain,
                            "severity": a.severity,
                            "message": a.message,
                        }
                        for a in alerts
                    ]
                }
            )

    console.print(
        f"[bold green]Starting Radar Mesh Node on port {port}...[/bold green]"
    )
    console.print(
        "[dim]Other devices on your network can sync from http://<your-ip>:{port}/api/sync[/dim]"
    )
    uvicorn.run(api, host="0.0.0.0", port=port, log_level="warning")


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
