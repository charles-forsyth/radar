import typer
import sys
import asyncio
import httpx
import click
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from google.cloud.sql.connector import Connector
from radar.core.ingest import (
    TextIngestAgent,
    WebIngestAgent,
    IntelligenceAgent,
    DeepResearchAgent,
    ADSBScanner,
    APRSStreamer,
)
from radar.config import settings
from radar.db.engine import set_global_connector
from radar.db.init import init_db
from radar.db.models import Entity, Connection, Signal, Trend, ChatSession, ChatMessage
from sqlalchemy import select
from sqlalchemy import func, desc

app = typer.Typer(name="radar", help="📡 Personal Industry Intelligence Brain")
console = Console()


@app.command()
def dashboard(
    refresh: int = typer.Option(
        5, "--refresh", "-r", help="Refresh interval in seconds."
    ),
):
    """Display a live intelligence dashboard in the terminal."""

    async def do_dashboard():
        from radar.db.engine import async_session
        from rich.live import Live
        import time

        # Connect to DB
        connector = None
        if settings.INSTANCE_CONNECTION_NAME:
            loop = asyncio.get_running_loop()
            connector = Connector(loop=loop)
            set_global_connector(connector)
            await connector.__aenter__()

        try:
            with Live(console=console, screen=True, auto_refresh=False) as live:
                while True:
                    async with async_session() as session:
                        # Fetch Stats
                        s_count = (
                            await session.execute(select(func.count(Signal.id)))
                        ).scalar()
                        e_count = (
                            await session.execute(select(func.count(Entity.id)))
                        ).scalar()
                        c_count = (
                            await session.execute(select(func.count(Connection.id)))
                        ).scalar()
                        t_count = (
                            await session.execute(select(func.count(Trend.id)))
                        ).scalar()

                        # Fetch Data
                        signals = (
                            (
                                await session.execute(
                                    select(Signal).order_by(desc(Signal.date)).limit(10)
                                )
                            )
                            .scalars()
                            .all()
                        )
                        trends = (
                            (
                                await session.execute(
                                    select(Trend).order_by(desc(Trend.id)).limit(5)
                                )
                            )
                            .scalars()
                            .all()
                        )
                        entities = (
                            (
                                await session.execute(
                                    select(Entity).order_by(func.random()).limit(15)
                                )
                            )
                            .scalars()
                            .all()
                        )

                        stats = {
                            "signals": s_count,
                            "entities": e_count,
                            "connections": c_count,
                            "trends": t_count,
                        }

                        from radar.ui.dashboard import render_dashboard_layout

                        layout = render_dashboard_layout(
                            signals, entities, trends, stats
                        )
                        live.update(layout, refresh=True)

                    time.sleep(refresh)

        except KeyboardInterrupt:
            pass
        finally:
            if connector:
                await connector.__aexit__(None, None, None)

    asyncio.run(do_dashboard())


@app.command()
def export(output: str = "radar_export.json"):
    """Export the entire Knowledge Graph to a JSON file."""

    async def do_export():
        from radar.db.engine import async_session
        from sqlalchemy import select

        # Connect to DB
        connector = None
        if settings.INSTANCE_CONNECTION_NAME:
            loop = asyncio.get_running_loop()
            connector = Connector(loop=loop)
            set_global_connector(connector)
            await connector.__aenter__()

        try:
            async with async_session() as session:
                console.print("[bold blue]Exporting Knowledge Graph...[/bold blue]")

                result_entities = await session.execute(select(Entity))
                entities = result_entities.scalars().all()

                result_connections = await session.execute(select(Connection))
                connections = result_connections.scalars().all()

                result_trends = await session.execute(select(Trend))
                trends = result_trends.scalars().all()

                data = {
                    "entities": [
                        {
                            "name": e.name,
                            "type": e.type,
                            "details": e.details,
                            "id": str(e.id),
                        }
                        for e in entities
                    ],
                    "connections": [
                        {
                            "source_uuid": str(c.source_uuid),
                            "target_uuid": str(c.target_uuid),
                            "type": c.type,
                            "meta_data": c.meta_data,
                        }
                        for c in connections
                    ],
                    "trends": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "velocity": t.velocity,
                        }
                        for t in trends
                    ],
                }

                import json

                with open(output, "w") as f:
                    json.dump(data, f, indent=4)

                console.print(f"[bold green]Exported to {output}[/bold green]")

        finally:
            if connector:
                await connector.__aexit__(None, None, None)

    asyncio.run(do_export())


@app.command()
def init():
    """Initialize the knowledge graph database."""

    async def do_init():
        if settings.INSTANCE_CONNECTION_NAME:
            loop = asyncio.get_running_loop()
            connector = Connector(loop=loop)
            set_global_connector(connector)
            async with connector:
                await init_db()
        else:
            await init_db()

    try:
        asyncio.run(do_init())
        console.print(
            Panel(
                "[bold green]📡 RADAR SYSTEMS INITIALIZED[/bold green]\nKnowledge Graph ready for signals."
            )
        )
    except Exception as e:
        console.print(f"[bold red]Initialization Failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command(name="list")
def list_signals():
    """List all ingested signals."""

    async def do_list():
        from radar.db.engine import async_session
        from sqlalchemy import select
        from rich.table import Table

        # Connect to DB
        connector = None
        if settings.INSTANCE_CONNECTION_NAME:
            loop = asyncio.get_running_loop()
            connector = Connector(loop=loop)
            set_global_connector(connector)
            await connector.__aenter__()

        try:
            async with async_session() as session:
                stmt = select(Signal).order_by(desc(Signal.date))
                result = await session.execute(stmt)
                signals = result.scalars().all()

                if not signals:
                    console.print("[yellow]No signals found.[/yellow]")
                    return

                table = Table(title="📡 Ingested Signals")
                table.add_column("Date", style="dim")
                table.add_column("Source", style="cyan")
                table.add_column("Title", style="bold white")

                for s in signals:
                    table.add_row(s.date.strftime("%Y-%m-%d %H:%M"), s.source, s.title)

                console.print(table)

        finally:
            if connector:
                await connector.__aexit__(None, None, None)

    asyncio.run(do_list())


@app.command()
def ask(
    question: Optional[str] = typer.Argument(None, help="The question to ask."),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Start an interactive chat session."
    ),
    session_id: Optional[str] = typer.Option(
        None, "--session", help="Continue an existing chat session ID."
    ),
    json_out: bool = typer.Option(
        False, "--json", help="Output the answer and sources in JSON format."
    ),
):
    """Ask a question or start an interactive chat based on ingested intelligence."""

    async def do_ask():
        from radar.db.engine import async_session
        from sqlalchemy import select
        import uuid
        import json

        intel = IntelligenceAgent()

        # Connect to DB
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
                # Create new session
                async with async_session() as session:
                    new_session = ChatSession(
                        title=question[:50] if question else "Interactive Session"
                    )
                    session.add(new_session)
                    await session.commit()
                    await session.refresh(new_session)
                    active_session_id = new_session.id
                    console.print(
                        f"[bold green]Started new session:[/bold green] {active_session_id}"
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
                    console.print(
                        "[red]Error: No question provided. Use 'radar ask [question]' or 'radar ask -i'.[/red]"
                    )
                    break

                # 1. Generate embedding for the question
                question_vector = await intel.get_embedding(current_question)

                # 2. Retrieve top-k relevant signals using vector similarity
                async with async_session() as session:
                    # Calculate distance and order by it
                    stmt = (
                        select(Signal)
                        .order_by(Signal.vector.l2_distance(question_vector))
                        .limit(5)
                    )
                    result = await session.execute(stmt)
                    relevant_signals = result.scalars().all()

                    # 3. Fetch history if in a session
                    history = []
                    if active_session_id:
                        hist_stmt = (
                            select(ChatMessage)
                            .where(ChatMessage.session_id == active_session_id)
                            .order_by(ChatMessage.created_at)
                        )
                        hist_result = await session.execute(hist_stmt)
                        history_msgs = hist_result.scalars().all()
                        for msg in history_msgs:
                            history.append({"role": msg.role, "content": msg.content})

                if not relevant_signals:
                    console.print(
                        "[yellow]No intelligence signals found in database.[/yellow]"
                    )
                    if not interactive:
                        break
                    current_question = None
                    continue

                # 4. Use Gemini to answer
                if json_out:
                    if active_session_id:
                        answer = await intel.chat(
                            current_question, relevant_signals, history
                        )
                    else:
                        answer = await intel.answer_question(
                            current_question, relevant_signals
                        )
                else:
                    with console.status("[bold blue]Thinking...[/bold blue]"):
                        if active_session_id:
                            answer = await intel.chat(
                                current_question, relevant_signals, history
                            )
                        else:
                            answer = await intel.answer_question(
                                current_question, relevant_signals
                            )

                # 5. Persist to history if in a session
                if active_session_id:
                    async with async_session() as session:
                        user_msg = ChatMessage(
                            session_id=active_session_id,
                            role="user",
                            content=current_question,
                        )
                        assistant_msg = ChatMessage(
                            session_id=active_session_id,
                            role="assistant",
                            content=answer,
                        )
                        session.add(user_msg)
                        session.add(assistant_msg)
                        await session.commit()

                # 6. Display the answer
                if json_out:
                    output_data = {
                        "question": current_question,
                        "answer": answer,
                        "sources": [
                            {"title": s.title, "url": s.url, "date": s.date.isoformat()}
                            for s in relevant_signals
                        ],
                    }
                    print(json.dumps(output_data, indent=2))
                else:
                    console.print(
                        Panel(
                            answer,
                            title=f"[bold cyan]Question: {current_question}[/bold cyan]",
                        )
                    )

                    # Show sources
                    sources = ", ".join(set([s.title for s in relevant_signals]))
                    console.print(f"[dim]Sources: {sources}[/dim]")

                if not interactive:
                    break

                current_question = None

        finally:
            if connector:
                await connector.__aexit__(None, None, None)

    asyncio.run(do_ask())


@app.command()
def live(
    refresh: int = typer.Option(
        2, "--refresh", "-r", help="Refresh interval in seconds."
    ),
):
    """Launch the Real-time Tactical SIGINT HUD."""

    async def do_live():
        from rich.live import Live
        from rich.table import Table
        from rich.layout import Layout
        from rich import box
        import asyncio

        scanner = ADSBScanner()
        aprs = APRSStreamer()

        # Start APRS in background
        asyncio.create_task(aprs.start_stream())

        def make_layout() -> Layout:
            layout = Layout()
            layout.split(
                Layout(name="header", size=3),
                Layout(name="body", ratio=1),
                Layout(name="footer", size=3),
            )
            layout["body"].split_column(
                Layout(name="adsb", ratio=1), Layout(name="aprs", ratio=1)
            )
            return layout

        def render_header() -> Panel:
            from datetime import datetime

            return Panel(
                f"[bold yellow]📡 RADAR LIVE[/bold yellow] | [cyan]Tactical SIGINT HUD[/cyan] | [dim]{datetime.now().strftime('%H:%M:%S')}[/dim]",
                style="white on blue",
                box=box.ROUNDED,
            )

        def render_aircraft_table(data: dict) -> Panel:
            table = Table(expand=True, box=box.SIMPLE_HEAD)
            table.add_column("ICAO", style="cyan", width=8)
            table.add_column("Flight", style="bold white", width=10)
            table.add_column("Altitude", justify="right", style="green")
            table.add_column("Speed", justify="right", style="yellow")
            table.add_column("Lat/Lon", style="dim")
            table.add_column("Msgs", justify="right", style="magenta")

            if "error" in data:
                return Panel(
                    f"[bold red]Sensor Error:[/bold red] {data['error']}",
                    title="ADS-B Status",
                    border_style="red",
                )

            aircraft_list = data.get("aircraft", [])
            for ac in aircraft_list:
                table.add_row(
                    ac.get("hex", "N/A"),
                    ac.get("flight", "Unk").strip(),
                    f"{ac.get('alt_baro', 0):,} ft",
                    f"{ac.get('gs', 0):.0f} kt",
                    f"{ac.get('lat', 0):.3f}, {ac.get('lon', 0):.3f}",
                    str(ac.get("messages", 0)),
                )

            return Panel(
                table,
                title=f"[bold green]ADS-B Aircraft Overhead ({len(aircraft_list)})[/bold green]",
                border_style="green",
            )

        def render_aprs_log(packets: list) -> Panel:
            table = Table(show_header=False, expand=True, box=None)
            table.add_column("Packet", style="white")

            # Show last 10 packets
            for p in packets[-10:]:
                table.add_row(p)

            return Panel(
                table,
                title="[bold blue]Live APRS Feed (Tioga County Sector)[/bold blue]",
                border_style="blue",
                subtitle="[dim]Global APRS-IS Stream[/dim]",
            )

        layout = make_layout()
        with Live(layout, screen=True, refresh_per_second=1):
            while True:
                data = await scanner.get_live_data()
                layout["header"].update(render_header())
                layout["adsb"].update(render_aircraft_table(data))
                layout["aprs"].update(render_aprs_log(aprs.packets))
                layout["footer"].update(
                    Panel(
                        "[dim]Press Ctrl+C to exit tactical view[/dim]", box=box.SIMPLE
                    )
                )
                await asyncio.sleep(refresh)

    try:
        asyncio.run(do_live())
    except KeyboardInterrupt:
        pass


@app.command()
def ingest(
    source: Optional[str] = typer.Argument(
        None, help="Source to ingest: a file path, URL, or '-' for stdin"
    ),
    file: Optional[typer.FileText] = typer.Option(
        None, "--file", "-f", help="Legacy file option (preferred: radar ingest [path])"
    ),
    voice: bool = typer.Option(
        False, "--voice", "-v", help="Enable voice confirmation."
    ),
):
    """Ingest massive textual intelligence via stdin, file, or URL."""

    async def _run_ingest_logic():
        from radar.db.engine import async_session

        # 1. Handle URL
        if source and (source.startswith("http://") or source.startswith("https://")):
            agent = WebIngestAgent()
            try:
                signal, kg = await agent.ingest(source)
            except httpx.HTTPStatusError as e_http:
                console.print(
                    f"[bold red]HTTP Error {e_http.response.status_code}:[/bold red] {e_http}"
                )
                if e_http.response.status_code == 403:
                    console.print(
                        "[yellow]Tip: The site might be blocking bots. We're using a standard browser User-Agent now, but some strict WAFs may still block us.[/yellow]"
                    )
                raise typer.Exit(code=1)
            except httpx.RequestError as e_req:
                console.print(f"[bold red]Network Error:[/bold red] {e_req}")
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
                except Exception as e_file:
                    console.print(
                        f"[bold red]Error reading file {source}:[/bold red] {e_file}"
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
            signal, kg = await agent.ingest(text)

        # 3. Persist to DB
        try:
            # 3a. Generate embeddings for extracted items
            items_to_embed = []
            for e in kg.entities:
                items_to_embed.append(f"{e.name}: {e.description}")
            for t in kg.trends:
                items_to_embed.append(f"{t.name}: {t.description}")

            # Batch call to Gemini
            if items_to_embed:
                vectors = await agent.intel.get_batch_embeddings(items_to_embed)
            else:
                vectors = []

            # Map back to objects
            entity_vectors = vectors[: len(kg.entities)]
            trend_vectors = vectors[len(kg.entities) :]

            async with async_session() as session:
                session.add(signal)
                await session.flush()  # Get ID for signal

                # Process Entities
                entity_map = {}  # Name -> UUID

                for idx, extracted_entity in enumerate(kg.entities):
                    # Check if entity exists
                    stmt = select(Entity).where(Entity.name == extracted_entity.name)
                    result = await session.execute(stmt)
                    existing_entity = result.scalar_one_or_none()

                    if existing_entity:
                        entity_map[extracted_entity.name] = existing_entity.id
                        # Optional: Update vector/description if needed?
                        # For now, let's keep the first one or assume stability.
                    else:
                        new_entity = Entity(
                            name=extracted_entity.name,
                            type=extracted_entity.type,
                            details={"description": extracted_entity.description},
                            vector=(
                                entity_vectors[idx]
                                if idx < len(entity_vectors)
                                else None
                            ),
                        )
                        session.add(new_entity)
                        await session.flush()
                        entity_map[extracted_entity.name] = new_entity.id

                # Process Trends
                for idx, extracted_trend in enumerate(kg.trends):
                    stmt = select(Trend).where(Trend.name == extracted_trend.name)
                    result = await session.execute(stmt)
                    existing_trend = result.scalar_one_or_none()

                    if not existing_trend:
                        new_trend = Trend(
                            name=extracted_trend.name,
                            description=extracted_trend.description,
                            velocity=extracted_trend.velocity,
                            vector=(
                                trend_vectors[idx] if idx < len(trend_vectors) else None
                            ),
                        )
                        session.add(new_trend)

                # Process Connections
                for extracted_conn in kg.connections:
                    source_id = entity_map.get(extracted_conn.source_entity_name)
                    target_id = entity_map.get(extracted_conn.target_entity_name)

                    if source_id and target_id:
                        new_conn = Connection(
                            source_uuid=source_id,
                            target_uuid=target_id,
                            type=extracted_conn.type,
                            meta_data={
                                "description": extracted_conn.description,
                                "signal_id": str(signal.id),
                            },
                        )
                        session.add(new_conn)

                await session.commit()
                await session.refresh(signal)
        except (OSError, asyncio.TimeoutError) as e_conn:
            console.print(
                f"[bold red]Database Connection Error:[/bold red] Could not connect to the database at {settings.DB_URL if not settings.INSTANCE_CONNECTION_NAME else settings.INSTANCE_CONNECTION_NAME}"
            )
            console.print(f"[dim]Error details: {e_conn}[/dim]")
            raise typer.Exit(code=1)
        except Exception as e_db:
            console.print(f"[bold red]Database Error:[/bold red] {e_db}")
            raise typer.Exit(code=1)

        return signal, kg

    async def process_ingestion():
        # Initialize Cloud SQL Connector if needed
        if settings.INSTANCE_CONNECTION_NAME:
            # Explicitly capture the running loop to prevent mismatch errors
            loop = asyncio.get_running_loop()

            # Use Connector as a context manager if possible, or manually managing scope
            # Here we initialize it inside the async loop with explicit loop argument
            connector = Connector(loop=loop)
            set_global_connector(connector)
            # Use it as an async context manager to ensure clean up
            async with connector:
                return await _run_ingest_logic()
        else:
            return await _run_ingest_logic()

    signal, kg = asyncio.run(process_ingestion())

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

    # KG Stats
    table.add_row("[bold magenta]Entities[/bold magenta]", str(len(kg.entities)))
    table.add_row("[bold magenta]Connections[/bold magenta]", str(len(kg.connections)))
    table.add_row("[bold magenta]Trends[/bold magenta]", str(len(kg.trends)))

    # Add a preview snippet
    snippet = (
        signal.content[:200].replace("\n", " ") + "..."
        if len(signal.content) > 200
        else signal.content
    )
    table.add_row("[bold cyan]Preview[/bold cyan]", f"[dim]{snippet}[/dim]")

    console.print(table)

    if voice:
        import subprocess

        voice_text = f"Signal ingested. I have identified {len(kg.entities)} entities and {len(kg.trends)} emerging trends from {signal.source}."
        subprocess.run(
            ["python3", "/home/chuck/Scripts/generate_voice.py", "--temp", voice_text]
        )


@app.command()
def sweep(
    topics_file: str = typer.Argument(
        ..., help="Path to a text file containing one research topic per line."
    ),
    voice: bool = typer.Option(
        False, "--voice", "-v", help="Enable voice confirmation per topic."
    ),
):
    """Autonomously research a list of topics using Gemini Deep Research and ingest them."""
    import os
    import subprocess

    if not os.path.exists(topics_file):
        console.print(f"[bold red]Error:[/bold red] File not found: {topics_file}")
        raise typer.Exit(code=1)

    with open(topics_file, "r") as f:
        topics = [line.strip() for line in f if line.strip()]

    console.print(f"[bold blue]Starting Sweep on {len(topics)} topics...[/bold blue]")

    agent = DeepResearchAgent()

    for i, topic in enumerate(topics, 1):
        console.print(
            f"\n[bold cyan]({i}/{len(topics)}) Researching:[/bold cyan] {topic}"
        )
        with console.status(
            f"[bold yellow]Deep Researching: {topic}... (This may take several minutes)[/bold yellow]"
        ):
            try:
                # Use asyncio.run for the async research method
                research_text = asyncio.run(agent.research(topic))

                # Prepend the topic as the title
                final_text = f"Title: Deep Research - {topic}\n\n{research_text}"

                # Re-use the ingest command via subprocess to handle all the DB and KG logic
                cmd = ["uv", "run", "radar", "ingest", "-"]
                if voice:
                    cmd.append("--voice")

                result = subprocess.run(
                    cmd, input=final_text, text=True, capture_output=True
                )

                if result.returncode == 0:
                    console.print(
                        f"[bold green]Successfully ingested research for:[/bold green] {topic}"
                    )
                else:
                    console.print(
                        f"[bold red]Failed to ingest research for {topic}:[/bold red]\n{result.stderr}"
                    )

            except Exception as e:
                console.print(f"[bold red]Research failed for {topic}:[/bold red] {e}")

    console.print("[bold green]Sweep Complete![/bold green]")
