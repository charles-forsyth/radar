import asyncio
import click
import typer
from datetime import datetime
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
    Signal,
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
                        .order_by(ChatMessage.timestamp)  # type: ignore
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
    """Helper to persist a signal and its extracted stats to SQLite."""
    # Extract structured statistics from content
    extracted_stats = intel.extract_stats(signal.content)

    async with async_session() as session:
        try:
            session.add(signal)
            await session.flush()

            # Save extracted stats
            for s in extracted_stats:
                # Categorize based on title keywords or label
                category = "GENERAL"
                t_lower = signal.title.lower()
                l_lower = s["label"].lower()

                if any(
                    k in t_lower
                    for k in [
                        "price",
                        "cost",
                        "economic",
                        "finance",
                        "route intel",
                        "gas",
                    ]
                ):
                    category = "FINANCE"
                elif any(
                    k in t_lower for k in ["benchmark", "performance", "fps", "t/s"]
                ):
                    category = "TECH_METRICS"
                elif any(
                    k in t_lower or k in l_lower
                    for k in [
                        "capacity",
                        "count",
                        "stats",
                        "troops",
                        "casualties",
                        "density",
                        "enforcement",
                    ]
                ):
                    category = "LOGISTICS/OSINT"
                elif any(
                    k in t_lower or k in l_lower
                    for k in [
                        "sdr",
                        "radio",
                        "p25",
                        "frequency",
                        "bandwidth",
                        "encryption",
                        "noise floor",
                        "rssi",
                        "sigint",
                    ]
                ):
                    category = "SIGINT/COMSEC"
                elif any(
                    k in t_lower
                    for k in [
                        "phishing",
                        "malware",
                        "vuln",
                        "cve",
                        "zero-day",
                        "exploit",
                    ]
                ):
                    category = "CYBER_INTEL"

                session.add(
                    Statistic(
                        category=category,
                        label=s["label"],
                        value=s["value"],
                        unit=s["unit"],
                        source_signal_id=signal.id,
                    )
                )
            await session.commit()
        except Exception as e:
            await session.rollback()
            if "duplicate key" not in str(e).lower():
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
    """Generate a high-fidelity 'SQLite Data Console' for pre-defined stats tracking."""
    import jinja2
    from sqlalchemy import select, desc
    import asyncio

    async def _report():
        console.print("[bold blue]Forging SQLite Tactical Data Console...[/bold blue]")

        async with async_session() as session:
            # 1. Fetch Relational Data
            tel_stmt = select(Telemetry).order_by(desc(Telemetry.timestamp)).limit(2)
            tel_results = (await session.execute(tel_stmt)).scalars().all()
            curr_tel = tel_results[0] if len(tel_results) > 0 else None

            river_stmt = (
                select(RiverLevel).order_by(desc(RiverLevel.timestamp)).limit(10)
            )
            river_results = (await session.execute(river_stmt)).scalars().all()

            rf_stmt = select(RFPeak).order_by(desc(RFPeak.timestamp)).limit(12)
            rf_peaks = (await session.execute(rf_stmt)).scalars().all()

            sw_stmt = (
                select(SoftwareInventory)
                .order_by(desc(SoftwareInventory.timestamp))
                .limit(4)
            )
            sw_inv = (await session.execute(sw_stmt)).scalars().all()

            alert_stmt = (
                select(TacticalAlert).order_by(desc(TacticalAlert.created_at)).limit(10)
            )
            alerts = (await session.execute(alert_stmt)).scalars().all()

            stats_stmt = (
                select(Statistic).order_by(desc(Statistic.timestamp)).limit(100)
            )
            all_stats = (await session.execute(stats_stmt)).scalars().all()

        # Group by category
        categorized_stats = {}
        for s in all_stats:
            if s.category not in categorized_stats:
                categorized_stats[s.category] = []
            if len(categorized_stats[s.category]) < 10:
                categorized_stats[s.category].append(s)

        report_css = """
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=JetBrains+Mono:wght@400;700&display=swap');
        body { background-color: #030508; color: #00ff41; font-family: 'JetBrains Mono', monospace; margin: 0; padding: 20px; text-transform: uppercase; font-size: 10px; letter-spacing: 0.5px; }
        .grid { display: grid; grid-template-columns: 350px 1fr 350px; gap: 15px; height: auto; }
        .header { grid-column: 1 / span 3; border: 2px solid #00ff41; padding: 15px; display: flex; justify-content: space-between; align-items: center; background: #00ff4111; margin-bottom: 20px; }
        .header-title { font-family: 'Orbitron', sans-serif; font-size: 1.5rem; font-weight: 900; }
        .box { border: 1px solid #00ff41; background: #080c12; padding: 15px; position: relative; margin-bottom: 15px; box-shadow: inset 0 0 10px #00ff4111; overflow-y: auto; }
        .box-label { position: absolute; top: -10px; left: 15px; background: #030508; border: 1px solid #00ff41; padding: 0 8px; color: #00ff41; font-weight: bold; font-size: 9px; }
        .big-val { font-size: 2.5rem; font-weight: 900; text-align: center; text-shadow: 0 0 15px #00ff41; margin: 5px 0; }
        .stat-row { display: flex; justify-content: space-between; border-bottom: 1px solid #002105; padding: 6px 0; }
        .stat-label { color: #008f11; }
        .cat-head { color: #000; background: #00ff41; padding: 2px 8px; font-weight: bold; font-size: 9px; margin-top: 15px; margin-bottom: 10px; display: inline-block; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: #00ff41; }
        """

        template = jinja2.Template("""
        <html>
        <head><style>{{ css }}</style><title>RADAR SQLITE CONSOLE</title></head>
        <body>
            <div class="header">
                <div class="header-title">RADAR // SQLITE TACTICAL CONSOLE</div>
                <div style="text-align: right; font-weight: bold;">SECTOR: TIOGA PA | {{ now }} | v{{ version }}</div>
            </div>
            
            <div class="grid">
                <div class="col">
                    <div class="box"><div class="box-label">ATMOSPHERICS</div><div class="big-val">{{ tel.temp_f }}°F</div></div>
                    <div class="box"><div class="box-label">AIRSPACE DENSITY</div><div class="big-val">{{ tel.aircraft_count }}</div></div>
                    <div class="box">
                        <div class="box-label">HYDROLOGY</div>
                        {% for r in rivers %}<div class="stat-row"><span class="stat-label">{{ r.station_name[:20] }}</span><span>{{ r.value }} {{ r.unit }}</span></div>{% endfor %}
                    </div>
                </div>

                <div class="col">
                    <div class="box" style="min-height: 500px;">
                        <div class="box-label">PRE-DEFINED STATISTICAL DATA WALL</div>
                        {% for cat, items in stats.items() %}
                        <div class="cat-head">// {{ cat }}</div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            {% for s in items %}
                            <div style="border: 1px solid #004111; padding: 8px; background: #0a0f18;">
                                <div style="font-size: 8px; color: #008f11; margin-bottom: 3px;">{{ s.label }}</div>
                                <div style="font-size: 1.1rem; font-weight: bold;">{{ s.value }} {{ s.unit }}</div>
                            </div>
                            {% endfor %}
                        </div>
                        {% endfor %}
                    </div>
                </div>

                <div class="col">
                    <div class="box">
                        <div class="box-label">SIGINT SPECTRUM</div>
                        {% for f in rf[:12] %}<div class="stat-row"><span class="stat-label">{{ f.frequency_mhz }} MHz</span><span>{{ f.power_db }} dB</span></div>{% endfor %}
                    </div>
                    <div class="box">
                        <div class="box-label">SYSTEM SOFTWARE</div>
                        {% for s in sw %}<div class="stat-row"><span class="stat-label">{{ s.manager.upper() }}</span><span>{{ s.package_count }} PKGS</span></div>{% endfor %}
                    </div>
                    <div class="box" style="border-color:#ff3131;">
                        <div class="box-label" style="color:#ff3131;border-color:#ff3131;">THREAT LOG</div>
                        {% for a in alerts[:5] %}<div style="color:#ff3131; font-size:9px; margin-bottom:5px; border-bottom:1px solid #ff313122;">! {{ a.message }}</div>{% endfor %}
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
            rivers=river_results,
            rf=rf_peaks,
            sw=sw_inv,
            stats=categorized_stats,
            alerts=alerts,
            now=now_str,
            version="0.36.0",
        )

        fname = "tactical_intelligence_briefing.html"
        with open(fname, "w") as f:
            f.write(html)
        console.print(f"[bold green]SQLite Console forged: {fname}[/bold green]")
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
    """Graph visualization disabled in v0.36 pivot."""
    console.print(
        "[yellow]Knowledge Graph visualization is disabled in the current metrics-first architecture.[/yellow]"
    )


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
    """Briefing engine disabled in v0.36 pivot."""
    console.print(
        "[yellow]Intelligence Briefing engine is disabled in the current metrics-first architecture.[/yellow]"
    )


@app.command()
def init():
    """Initialize the local SQLite database and verify table integrity."""

    async def _init():
        console.print(
            "[bold blue]RADAR SYSTEM INITIALIZATION SEQUENCE STARTED[/bold blue]"
        )
        await init_db()
        console.print("[bold green]RADAR SYSTEM READY - DATABASE ONLINE[/bold green]")

    asyncio.run(_init())


if __name__ == "__main__":
    app()
