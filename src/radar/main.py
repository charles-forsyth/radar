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
from radar.core.models import KnowledgeGraphExtraction
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
from radar.config import settings


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

                    # Process topics concurrently
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
                        roam_cmd = [
                            settings.ROAM_BIN,
                            "route",
                            place,
                            "--weather",
                            "-F",
                            "gas",
                        ]
                        result = subprocess.run(
                            roam_cmd, capture_output=True, text=True
                        )
                        if result.returncode == 0:
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
                                url, inst = parts[0].strip(), parts[1].strip()
                                console.print(f"[cyan]Dynamic Scrape:[/cyan] {url}")
                                try:
                                    text = await browser_agent.extract(url, inst)
                                    final_text = f"Title: Dynamic Web Extraction - {url}\n\n{text}"
                                    await run_ingest(final_text, voice, shared_intel)
                                except Exception as e:
                                    console.print(
                                        f"[red]Dynamic Scrape failed for {url}:[/red] {e}"
                                    )

            if tactical:
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
                        .order_by(desc(Signal.date))  # type: ignore
                        .limit(50)
                    )
                    baseline_sitreps = (
                        (await session.execute(baseline_stmt)).scalars().all()
                    )

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
                        severity, domain, msg = (
                            anomaly["severity"],
                            anomaly["domain"],
                            anomaly["message"],
                        )
                        session.add(
                            TacticalAlert(domain=domain, severity=severity, message=msg)
                        )

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
                                        settings.PYTHON_BIN,
                                        settings.VOICE_SCRIPT,
                                        "--temp",
                                        alert_text,
                                    ]
                                )
                        else:
                            console.print(
                                f"[bold yellow]Tactical Note [{domain}]:[/bold yellow] {msg}"
                            )

                    await session.commit()

                await run_ingest(sitrep_text, voice, shared_intel)

        finally:
            pass

    asyncio.run(do_sync())


async def save_ingest_to_db(
    signal: Signal, kg: KnowledgeGraphExtraction, intel: IntelligenceAgent
):
    """Helper to persist a signal and its extracted stats to SQLite."""
    extracted_stats = intel.extract_stats(signal.content)

    async with async_session() as session:
        try:
            session.add(signal)
            await session.flush()

            for s in extracted_stats:
                category = "GENERAL"
                t_lower, l_lower = signal.title.lower(), s["label"].lower()

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
                        description=s.get("description"),
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
            intel = shared_intel if shared_intel else agent.intel
            await save_ingest_to_db(signal, kg, intel)
            console.print(f"[green]Ingested:[/green] {signal.title}")

            if voice:
                alert_text = f"New intelligence report ingested: {signal.title}"
                subprocess.run(
                    [settings.PYTHON_BIN, settings.VOICE_SCRIPT, "--temp", alert_text]
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

    text = ""
    if source == "-":
        text = sys.stdin.read()
    elif source:
        try:
            with open(source, "r") as f:
                text = f.read()
        except Exception:
            raise typer.Exit(code=1)
    elif file:
        text = file.read()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        pass

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
    """Generate the v0.45.0 Unified Intelligence HUD (Map + Data Wall)."""
    import jinja2
    from sqlalchemy import select, desc
    import asyncio
    import folium
    import re
    import base64

    async def _generate_map_base64():
        tioga_coords = settings.HOME_COORDS
        m = folium.Map(location=tioga_coords, zoom_start=8, tiles="CartoDB dark_matter")

        folium.Circle(
            radius=settings.SECTOR_RADIUS_MILES * 1609.34,
            location=tioga_coords,
            popup=f"{settings.SECTOR_RADIUS_MILES}-Mile Sector",
            color="#00ff41",
            fill=True,
            fill_color="#00ff41",
            fill_opacity=0.05,
        ).add_to(m)

        folium.Marker(
            tioga_coords,
            popup="BASE",
            icon=folium.Icon(color="green", icon="home"),
        ).add_to(m)

        async with async_session() as session:
            stmt = select(Signal).where(Signal.title.contains("SITREP")).limit(100)
            sitreps = (await session.execute(stmt)).scalars().all()
            for s in sitreps:
                matches = re.finditer(
                    r"Lat:\s*([\-\d\.]+),\s*Lon:\s*([\-\d\.]+)", s.content
                )
                for match in matches:
                    lat, lon = float(match.group(1)), float(match.group(2))
                    folium.CircleMarker(
                        [lat, lon],
                        radius=4,
                        color="#ff3131",
                        fill=True,
                        popup="AIRCRAFT PING",
                    ).add_to(m)

        map_html = m._repr_html_()
        return base64.b64encode(map_html.encode()).decode()

    async def _report():
        console.print(
            "[bold blue]Forging v0.45 Unified Intelligence HUD...[/bold blue]"
        )
        map_b64 = await _generate_map_base64()

        async with async_session() as session:
            tel_stmt = select(Telemetry).order_by(desc(Telemetry.timestamp)).limit(1)  # type: ignore
            curr_tel = (await session.execute(tel_stmt)).scalar_one_or_none()

            river_stmt = (
                select(RiverLevel).order_by(desc(RiverLevel.timestamp)).limit(100)
            )  # type: ignore
            river_results = (await session.execute(river_stmt)).scalars().all()
            river_map = {}
            for r in river_results:
                if r.station_name not in river_map:
                    river_map[r.station_name] = []
                if len(river_map[r.station_name]) < 2:
                    river_map[r.station_name].append(r)
            latest_rivers = []
            for name, items in river_map.items():
                c, p = items[0], (items[1] if len(items) > 1 else None)
                latest_rivers.append(
                    {
                        "name": name,
                        "val": c.value,
                        "unit": c.unit,
                        "delta": c.value - p.value if p else 0,
                    }
                )

            rf_stmt = select(RFPeak).order_by(desc(RFPeak.timestamp)).limit(40)  # type: ignore
            rf_results = (await session.execute(rf_stmt)).scalars().all()
            rf_map = {}
            for r in rf_results:
                f_key = round(r.frequency_mhz, 1)
                if f_key not in rf_map:
                    rf_map[f_key] = []
                if len(rf_map[f_key]) < 2:
                    rf_map[f_key].append(r)
            processed_rf = []
            for f_key, items in rf_map.items():
                c, p = items[0], (items[1] if len(items) > 1 else None)
                processed_rf.append(
                    {
                        "freq": c.frequency_mhz,
                        "db": c.power_db,
                        "delta": c.power_db - p.power_db if p else 0,
                    }
                )
            processed_rf.sort(key=lambda x: x["db"], reverse=True)

            sw_stmt = (
                select(SoftwareInventory)
                .order_by(desc(SoftwareInventory.timestamp))
                .limit(20)
            )  # type: ignore
            sw_results = (await session.execute(sw_stmt)).scalars().all()
            sw_map = {}
            for s in sw_results:
                if s.manager not in sw_map:
                    sw_map[s.manager] = []
                if len(sw_map[s.manager]) < 2:
                    sw_map[s.manager].append(s)
            latest_sw = []
            for m_name, items in sw_map.items():
                c, p = items[0], (items[1] if len(items) > 1 else None)
                latest_sw.append(
                    {
                        "manager": m_name.upper(),
                        "count": c.package_count,
                        "delta": c.package_count - p.package_count if p else 0,
                    }
                )

            stats_stmt = (
                select(Statistic).order_by(desc(Statistic.timestamp)).limit(2000)
            )  # type: ignore
            all_stats = (await session.execute(stats_stmt)).scalars().all()

            alert_stmt = (
                select(TacticalAlert).order_by(desc(TacticalAlert.created_at)).limit(15)
            )  # type: ignore
            alerts = (await session.execute(alert_stmt)).scalars().all()

        stats_map = {}
        for s in all_stats:
            key = (s.category, s.label)
            if key not in stats_map:
                stats_map[key] = []
            if len(stats_map[key]) < 2:
                stats_map[key].append(s)

        final_stats = {}
        for (cat, label), items in stats_map.items():
            c, p = items[0], (items[1] if len(items) > 1 else None)
            if cat not in final_stats:
                final_stats[cat] = []
            final_stats[cat].append(
                {
                    "label": label,
                    "val": c.value,
                    "unit": c.unit or "",
                    "delta": c.value - p.value if p else 0,
                    "description": c.description,
                }
            )

        report_css = """
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&family=JetBrains+Mono:wght@400;700&display=swap');
        :root { --neon-green: #00ff41; --deep-bg: #020406; --glass-bg: rgba(8, 12, 18, 0.85); --alert-red: #ff3131; }
        body { background: var(--deep-bg); color: var(--neon-green); font-family: 'JetBrains Mono', monospace; margin: 0; padding: 15px; text-transform: uppercase; font-size: 10px; overflow-x: hidden; }
        .hud-grid { display: grid; grid-template-columns: 320px 1fr 320px; gap: 12px; height: calc(100vh - 80px); }
        .header { border: 1px solid var(--neon-green); padding: 12px 25px; display: flex; justify-content: space-between; align-items: center; background: rgba(0, 255, 65, 0.08); margin-bottom: 12px; border-radius: 4px; }
        .header-title { font-family: 'Orbitron', sans-serif; font-size: 1.4rem; font-weight: 900; text-shadow: 0 0 15px var(--neon-green); }
        .box { border: 1px solid var(--neon-green); background: var(--glass-bg); padding: 15px; position: relative; margin-bottom: 12px; border-radius: 4px; box-shadow: inset 0 0 10px rgba(0, 255, 65, 0.1); overflow-y: auto; }
        .box-label { position: absolute; top: -10px; left: 15px; background: var(--deep-bg); border: 1px solid var(--neon-green); padding: 2px 10px; color: var(--neon-green); font-weight: 900; font-size: 9px; }
        .center-stack { display: flex; flex-direction: column; gap: 12px; height: 100%; }
        .map-frame { border: 1px solid var(--neon-green); border-radius: 4px; height: 450px; width: 100%; position: relative; overflow: hidden; }
        .stat-row { display: flex; justify-content: space-between; border-bottom: 1px solid rgba(0, 255, 65, 0.1); padding: 5px 0; }
        .metric-big { font-size: 2.5rem; font-weight: 900; text-align: center; margin: 15px 0; font-family: 'Orbitron'; color: #fff; text-shadow: 0 0 10px var(--neon-green); }
        .data-card { border: 1px solid rgba(0, 255, 65, 0.2); background: rgba(5, 8, 12, 0.6); padding: 10px; margin-bottom: 10px; border-radius: 4px; transition: all 0.3s; }
        .data-card:hover { border-color: var(--neon-green); box-shadow: 0 0 15px rgba(0, 255, 65, 0.2); transform: translateY(-2px); }
        .stat-desc { text-transform: none; font-size: 10px; color: #8b949e; line-height: 1.5; margin-top: 8px; border-top: 1px solid rgba(0, 255, 65, 0.1); padding-top: 8px; font-style: italic; }
        .trend-up { color: var(--alert-red); } .trend-down { color: #39d353; }
        .cat-tag { color: #000; background: var(--neon-green); padding: 2px 8px; font-weight: 900; font-size: 9px; margin: 15px 0 10px 0; display: inline-block; }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-thumb { background: var(--neon-green); }
        .pulse { animation: pulse 2s infinite; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.3; } 100% { opacity: 1; } }
        .scan-line { width: 100%; height: 2px; background: rgba(0, 255, 65, 0.3); position: absolute; top: 0; left: 0; animation: scan 4s linear infinite; z-index: 100; pointer-events: none; }
        @keyframes scan { from { top: 0; } to { top: 100%; } }
        """

        template = jinja2.Template("""
        <!DOCTYPE html>
        <html>
        <head><style>{{ css }}</style><title>RADAR // UNIFIED COMMAND HUD</title></head>
        <body>
            <div class="header">
                <div class="header-title"><span class="pulse">●</span> RADAR COMMAND // UNIFIED INTELLIGENCE HUD</div>
                <div style="text-align: right; font-weight: 900;">
                    SECTOR: TIOGA PA | {{ now }} | v{{ version }}
                </div>
            </div>
            
            <div class="hud-grid">
                <!-- LEFT COLUMN: ATMOSPHERICS & SECURITY -->
                <div class="col">
                    <div class="box">
                        <div class="box-label">ENVIRONMENTAL SENSORS</div>
                        <div class="metric-big">{{ tel.temp_f if tel else '--' }}°F</div>
                    </div>
                    <div class="box">
                        <div class="box-label">NETWORK TOPOLOGY</div>
                        <div class="stat-row"><span>ACTIVE NODES</span><span>{{ tel.lan_device_count if tel else '0' }}</span></div>
                        <div class="stat-row"><span>SSH THREATS</span><span style="color:var(--alert-red)">{{ tel.ssh_failure_count if tel else '0' }}</span></div>
                        <div class="stat-row"><span>NET LATENCY</span><span>{{ tel.internet_latency_ms if tel else '--' }} MS</span></div>
                    </div>
                    <div class="box" style="height: 300px;">
                        <div class="box-label">HYDROLOGY Board</div>
                        {% for r in rivers %}
                        <div class="stat-row">
                            <span style="color:#008f11">{{ r.name[:20] }}</span>
                            <span>{{ r.val }} {{ r.unit }} <span style="color:#ffff00">({% if r.delta > 0 %}+{% endif %}{{ "%.2f"|format(r.delta) }})</span></span>
                        </div>
                        {% endfor %}
                    </div>
                    <div class="box" style="border-color: var(--alert-red);">
                        <div class="box-label" style="color:var(--alert-red); border-color:var(--alert-red);">ACTIVE THREAT LOG</div>
                        {% for a in alerts[:6] %}
                        <div style="color:var(--alert-red); font-size:9px; margin-bottom:5px; border-bottom:1px solid rgba(255,49,49,0.1); padding-bottom:3px;">
                            >> {{ a.message }}
                        </div>
                        {% endfor %}
                    </div>
                </div>

                <!-- CENTER COLUMN: TACTICAL MAP & STRATEGIC WALL -->
                <div class="col center-stack">
                    <div class="map-frame">
                        <div class="box-label">TACTICAL SITUATION MAP (LIVE)</div>
                        <div class="scan-line"></div>
                        <iframe src="data:text/html;base64,{{ map_data }}" style="width:100%; height:100%; border:none;"></iframe>
                    </div>
                    <div class="box" style="flex-grow: 1;">
                        <div class="box-label">STRATEGIC DATA WALL // TOTAL EXTRACTED OSINT/SIGINT</div>
                        {% for cat, items in stats.items() %}
                        <div class="cat-tag">// SECTOR: {{ cat }}</div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                            {% for s in items %}
                            <div class="data-card">
                                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                                    <div style="font-size: 11px; color: var(--neon-green); font-weight: 900;">{{ s.label }}</div>
                                    <div style="font-size: 1.2rem; font-weight: 900; text-align: right; color: #fff;">
                                        {{ s.val }} {{ s.unit }}
                                        {% if s.delta != 0 %}
                                        <span class="{% if s.delta > 0 %}trend-up{% else %}trend-down{% endif %}" style="font-size: 10px;">
                                            ({% if s.delta > 0 %}+{% endif %}{{ "%.2f"|format(s.delta) }})
                                        </span>
                                        {% endif %}
                                    </div>
                                </div>
                                {% if s.description %}<div class="stat-desc">{{ s.description }}</div>{% endif %}
                            </div>
                            {% endfor %}
                        </div>
                        {% endfor %}
                    </div>
                </div>

                <!-- RIGHT COLUMN: SIGINT & SYSTEM -->
                <div class="col">
                    <div class="box">
                        <div class="box-label">AIRSPACE DENSITY</div>
                        <div class="metric-big">{{ tel.aircraft_count if tel else '0' }}</div>
                    </div>
                    <div class="box" style="height: 400px;">
                        <div class="box-label">SIGINT SPECTRUM ANALYSIS</div>
                        {% for f in rf %}
                        <div class="stat-row">
                            <span style="color:#008f11">{{ f.freq }} MHZ</span>
                            <span>{{ f.db }} DB <span style="color:#ffff00">({% if f.delta > 0 %}+{% endif %}{{ "%.2f"|format(f.delta) }})</span></span>
                        </div>
                        {% endfor %}
                    </div>
                    <div class="box">
                        <div class="box-label">SOFTWARE ARSENAL</div>
                        {% for s in sw %}
                        <div class="stat-row">
                            <span style="color:#008f11">{{ s.manager }}</span>
                            <span>{{ s.count }} PKGS <span style="color:#ffff00">({% if s.delta > 0 %}+{% endif %}{{ s.delta }})</span></span>
                        </div>
                        {% endfor %}
                    </div>
                    <div class="box" style="border-color: #008f11; color: #008f11;">
                        <div class="box-label" style="border-color: #008f11;">SYSTEM CORE HEALTH</div>
                        <div style="font-size: 9px; line-height:1.6;">
                            CORE UPTIME: 312H 14M<br>
                            DB STATE: SQLITE RELATIONAL<br>
                            INTEGRITY: 100% NOMINAL
                        </div>
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
            rivers=latest_rivers,
            rf=processed_rf,
            sw=latest_sw,
            stats=final_stats,
            alerts=alerts,
            now=now_str,
            version="0.45.0",
            map_data=map_b64,
        )

        with open("tactical_intelligence_briefing.html", "w") as f:
            f.write(html)

        console.print(
            "[bold green]Unified Intelligence HUD forged: tactical_intelligence_briefing.html[/bold green]"
        )
        if open_browser:
            import subprocess

            try:
                subprocess.run(
                    ["xdg-open", "tactical_intelligence_briefing.html"], check=False
                )
            except Exception:
                pass

    asyncio.run(_report())


@app.command()
def stats(
    category: Optional[str] = typer.Option(None, help="Filter by category."),
    limit: int = typer.Option(100, help="Limit output results."),
):
    """View all extracted statistics with their full context in the terminal."""
    from rich.table import Table

    async def _stats():
        async with async_session() as session:
            stmt = select(Statistic).order_by(desc(Statistic.timestamp))  # type: ignore
            if category:
                stmt = stmt.where(Statistic.category == category.upper())  # type: ignore
            stmt = stmt.limit(limit)

            results = (await session.execute(stmt)).scalars().all()
            if not results:
                console.print("[yellow]No statistics found.[/yellow]")
                return

            table = Table(
                title="[bold green]RADAR EXTRACTED INTELLIGENCE LEDGER[/bold green]"
            )
            table.add_column("TIMESTAMP", style="dim")
            table.add_column("CATEGORY", style="bold cyan")
            table.add_column("LABEL", style="bold white")
            table.add_column("VALUE", justify="right", style="bold green")
            table.add_column("CONTEXT", style="dim", max_width=60)

            for s in results:
                table.add_row(
                    s.timestamp.strftime("%Y-%m-%d %H:%M"),
                    s.category,
                    s.label,
                    f"{s.value} {s.unit or ''}",
                    s.description or "N/A",
                )
            console.print(table)

    asyncio.run(_stats())


@app.command()
def map():
    """Generate an offline HTML map of extracted tactical coordinates."""
    import folium
    from sqlalchemy import select
    import re

    async def _map():
        console.print("[bold blue]Generating Offline Tactical Map...[/bold blue]")
        tioga_coords = settings.HOME_COORDS
        m = folium.Map(location=tioga_coords, zoom_start=7, tiles="CartoDB positron")

        folium.Circle(
            radius=settings.SECTOR_RADIUS_MILES * 1609.34,
            location=tioga_coords,
            popup=f"{settings.SECTOR_RADIUS_MILES}-Mile Strategic Sector",
            color="#3186cc",
            fill=True,
            fill_color="#3186cc",
            fill_opacity=0.1,
        ).add_to(m)

        folium.Marker(
            tioga_coords,
            popup="1539 Button Hill Road (Home Base)",
            icon=folium.Icon(color="green", icon="home"),
        ).add_to(m)

        async with async_session() as session:
            stmt = select(Signal).where(Signal.title.contains("SITREP")).limit(100)
            sitreps = (await session.execute(stmt)).scalars().all()

            for s in sitreps:
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
            )  # type: ignore
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
            )  # type: ignore
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
