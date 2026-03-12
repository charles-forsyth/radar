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
                sitrep_text = await agent.generate_full_sitrep(prev=prev_sitrep)

                # --- ANOMALY DETECTION ---
                with console.status(
                    "[bold magenta]Tactical Sentinel is analyzing for anomalies...[/bold magenta]"
                ):
                    anomalies = await shared_intel.detect_anomalies(
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

                            # System Desktop Notification
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
    """Generate an elite, tactical-aesthetic intelligence briefing with deep analytical synthesis."""
    import jinja2
    from sqlalchemy import select, desc
    import re

    async def _report():
        console.print(
            "[bold blue]Synthesizing Command Center Intelligence Briefing...[/bold blue]"
        )

        async with async_session() as session:
            sitrep_stmt = (
                select(Signal)
                .where(Signal.title.contains("SITREP"))
                .order_by(desc(Signal.date))
                .limit(2)
            )
            sitreps = (await session.execute(sitrep_stmt)).scalars().all()
            current_sitrep = sitreps[0] if len(sitreps) > 0 else None
            prev_sitrep = sitreps[1] if len(sitreps) > 1 else None

            research_stmt = (
                select(Signal)
                .where(Signal.title.contains("Deep Research"))
                .order_by(desc(Signal.date))
                .limit(12)
            )
            research_signals = (await session.execute(research_stmt)).scalars().all()

            alert_stmt = (
                select(TacticalAlert)
                .where(TacticalAlert.severity.in_(["WARNING", "CRITICAL"]))
                .order_by(desc(TacticalAlert.created_at))
                .limit(8)
            )
            alerts = (await session.execute(alert_stmt)).scalars().all()

        def clean_research(text):
            text = re.sub(
                r"Autonomous Research Report for:.*?\n", "", text, flags=re.IGNORECASE
            )
            text = re.sub(r"🎯.*?\n", "", text)
            text = re.sub(r"Question:.*?\n", "", text, flags=re.IGNORECASE)
            text = re.sub(r"Target:.*?\n", "", text, flags=re.IGNORECASE)
            text = re.sub(r"📌 http.*?\n", "", text)
            paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 40]
            return "\n\n".join(paras[:4])

        def parse_metrics(content):
            m = {"temp": "N/A", "devices": 0, "planes": 0, "ssh": 0, "rf_spike": "None"}
            t = re.search(r"(\d+\.\d+)°F", content)
            if t:
                m["temp"] = t.group(1)
            d = re.search(r"Local LAN Devices \(ARP\):\*\* (\d+)", content)
            if d:
                m["devices"] = int(d.group(1))
            m["planes"] = len(re.findall(r"Flight ", content))
            s = re.search(r"Failed SSH Logins \(Auth\):\*\* (\d+)", content)
            if s:
                m["ssh"] = int(s.group(1))
            rf = re.search(r"Frequency: ([\d\.]+) MHz \| Power: ([\d\.]+) dB", content)
            if rf:
                m["rf_spike"] = f"{rf.group(1)}MHz"
            return m

        curr_m = parse_metrics(current_sitrep.content) if current_sitrep else {}
        prev_m = parse_metrics(prev_sitrep.content) if prev_sitrep else {}

        deltas = []
        if curr_m and prev_m:
            if curr_m["devices"] != prev_m["devices"]:
                diff = curr_m["devices"] - prev_m["devices"]
                deltas.append(
                    f"Subnet Delta: {'+' if diff > 0 else ''}{diff} nodes detected."
                )
            if curr_m["ssh"] > prev_m["ssh"]:
                deltas.append(
                    f"Intrusion Escalation: SSH failures increased by {curr_m['ssh'] - prev_m['ssh']}."
                )
            if abs(curr_m["planes"] - prev_m["planes"]) > 3:
                deltas.append(
                    f"Air Traffic Shift: {curr_m['planes']} active pings in sector."
                )

        categories = {
            "TACTICAL TECH": [],
            "INFRASTRUCTURE": [],
            "SURVIVAL": [],
            "STRATEGIC": [],
        }
        for s in research_signals:
            body = clean_research(s.content)
            entry = {
                "title": s.title.replace("Deep Research - ", "").upper(),
                "body": body,
                "date": s.date.strftime("%H:%M"),
            }
            if any(
                k in s.title.lower()
                for k in ["cyber", "sdr", "radio", "p25", "encryption"]
            ):
                categories["TACTICAL TECH"].append(entry)
            elif any(
                k in s.title.lower() for k in ["tioga", "river", "flood", "emergency"]
            ):
                categories["INFRASTRUCTURE"].append(entry)
            elif any(k in s.title.lower() for k in ["foraging", "survival", "battery"]):
                categories["SURVIVAL"].append(entry)
            else:
                categories["STRATEGIC"].append(entry)

        report_css = """
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
        body { background-color: #05070a; color: #00ff41; font-family: 'JetBrains Mono', monospace; margin: 0; padding: 20px; text-transform: uppercase; font-size: 11px; letter-spacing: 1px; }
        .hud-grid { display: grid; grid-template-columns: 300px 1fr; gap: 20px; }
        .sidebar { border: 1px solid #00ff41; padding: 20px; box-shadow: inset 0 0 10px #00ff4133; height: calc(100vh - 80px); position: sticky; top: 20px; overflow-y: auto; }
        .main-view { border: 1px solid #00ff41; padding: 30px; box-shadow: inset 0 0 15px #00ff4122; }
        h1 { font-size: 1.8rem; text-align: center; border-bottom: 2px solid #00ff41; padding-bottom: 10px; margin-top: 0; text-shadow: 0 0 10px #00ff41; }
        .metric { border-bottom: 1px solid #004111; padding: 10px 0; display: flex; justify-content: space-between; }
        .metric-label { color: #008f11; }
        .metric-value { font-weight: bold; color: #00ff41; }
        .alert-row { border: 1px solid #ff3131; color: #ff3131; padding: 10px; margin-bottom: 10px; background: #310000; font-weight: bold; }
        .delta-row { color: #ffff00; font-size: 11px; margin-bottom: 5px; }
        .dispatch { border: 1px solid #004111; padding: 20px; margin-bottom: 25px; background: #080c12; }
        .dispatch-header { display: flex; justify-content: space-between; border-bottom: 1px solid #008f11; padding-bottom: 5px; margin-bottom: 10px; color: #00ff41; font-weight: bold; }
        .dispatch-body { color: #a0a0a0; text-transform: none; line-height: 1.4; white-space: pre-wrap; font-size: 12px; }
        .cat-head { color: #00ff41; background: #004111; padding: 5px 15px; display: inline-block; margin-bottom: 15px; }
        ::-webkit-scrollbar { width: 5px; }
        ::-webkit-scrollbar-thumb { background: #00ff41; }
        """

        template = jinja2.Template("""
        <html>
        <head><style>{{ css }}</style><title>RADAR COMMAND CENTER</title></head>
        <body>
            <div class="hud-grid">
                <div class="sidebar">
                    <h1>STATUS HUD</h1>
                    <div class="metric"><span class="metric-label">TEMP</span><span class="metric-value">{{ m.temp }}°F</span></div>
                    <div class="metric"><span class="metric-label">AIRCRAFT</span><span class="metric-value">{{ m.planes }} UNITS</span></div>
                    <div class="metric"><span class="metric-label">LAN NODES</span><span class="metric-value">{{ m.devices }}</span></div>
                    <div class="metric"><span class="metric-label">SSH FAIL</span><span class="metric-value">{{ m.ssh }}</span></div>
                    <div class="metric"><span class="metric-label">RF PEAK</span><span class="metric-value">{{ m.rf_spike }}</span></div>
                    
                    <h2 style="font-size: 14px; margin-top: 30px;">DELTA ANALYSIS</h2>
                    {% for d in deltas %}<div class="delta-row">>> {{ d }}</div>{% endfor %}

                    <h2 style="font-size: 14px; margin-top: 30px;">SYSTEM ALERTS</h2>
                    {% for a in alerts %}<div class="alert-row">! {{ a.severity }}: {{ a.message }}</div>{% endfor %}
                </div>

                <div class="main-view">
                    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #00ff41; margin-bottom: 30px;">
                        <span style="font-size: 2rem; font-weight: bold;">RADAR INTELLIGENCE BRIEFING</span>
                        <span style="text-align: right; color: #008f11;">SECTOR: TIOGA PA<br>GEN: {{ now }}</span>
                    </div>

                    {% for cat, items in categories.items() %}{% if items %}
                    <div class="cat-head">// SOURCE: {{ cat }}</div>
                    {% for item in items %}
                    <div class="dispatch">
                        <div class="dispatch-header"><span>>> {{ item.title }}</span><span>TIMESTAMP: {{ item.date }}</span></div>
                        <div class="dispatch-body">{{ item.body }}</div>
                    </div>
                    {% endfor %}{% endif %}{% endfor %}
                    
                    <div style="text-align: center; border-top: 1px solid #004111; padding-top: 20px; color: #004111; font-size: 10px;">
                        LOCAL SOVEREIGNTY ENGINE v{{ version }} | ENCRYPTED OFFLINE STORAGE | END OF DISPATCH
                    </div>
                </div>
            </div>
        </body>
        </html>
        """)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        html = template.render(
            css=report_css,
            m=curr_m,
            deltas=deltas,
            alerts=alerts,
            categories=categories,
            now=now_str,
            version="0.21.0",
        )

        fname = "tactical_intelligence_briefing.html"
        with open(fname, "w") as f:
            f.write(html)
        console.print(
            f"[bold green]Command Center Briefing generated at: {fname}[/bold green]"
        )
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
