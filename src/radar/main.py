import typer
import sys
import asyncio
import httpx
import click
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from google.cloud.sql.connector import Connector
from radar.core.ingest import TextIngestAgent, WebIngestAgent, IntelligenceAgent
from radar.config import settings
from radar.db.engine import set_global_connector
from radar.db.init import init_db
from radar.db.models import Entity, Connection, Signal, Trend
from sqlalchemy import select

app = typer.Typer(name="radar", help="📡 Personal Industry Intelligence Brain")
console = Console()


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


@app.command()
def scan(url: str):
    """Ingest a market signal from a URL."""
    console.print(f"[bold cyan]Scanning Signal:[/bold cyan] {url}")


@app.command()
def note(text: str):
    """Log a manual strategic observation."""
    console.print(f"[bold yellow]Strategic Note Recorded:[/bold yellow] {text}")


@app.command()
def ask(question: str):
    """Ask a question based on ingested intelligence."""

    async def do_ask():
        from radar.db.engine import async_session
        from sqlalchemy import select

        intel = IntelligenceAgent()

        # Connect to DB
        connector = None
        if settings.INSTANCE_CONNECTION_NAME:
            loop = asyncio.get_running_loop()
            connector = Connector(loop=loop)
            set_global_connector(connector)
            await connector.__aenter__()

        try:
            # 1. Generate embedding for the question
            question_vector = await intel.get_embedding(question)

            # 2. Retrieve top-k relevant signals using vector similarity
            async with async_session() as session:
                # Calculate distance and order by it
                # <-> operator is L2 distance in pgvector
                stmt = (
                    select(Signal)
                    .order_by(Signal.vector.l2_distance(question_vector))
                    .limit(5)
                )
                result = await session.execute(stmt)
                relevant_signals = result.scalars().all()

            if not relevant_signals:
                console.print(
                    "[yellow]No intelligence signals found in database.[/yellow]"
                )
                return

            # 3. Use Gemini to answer based on these signals
            with console.status("[bold blue]Thinking...[/bold blue]"):
                answer = await intel.answer_question(question, relevant_signals)

            # 4. Display the answer
            console.print(
                Panel(answer, title=f"[bold cyan]Question: {question}[/bold cyan]")
            )

            # Show sources
            sources = ", ".join(set([s.title for s in relevant_signals]))
            console.print(f"[dim]Sources: {sources}[/dim]")

        finally:
            if connector:
                await connector.__aexit__(None, None, None)

    asyncio.run(do_ask())


@app.command()
def ingest(
    source: Optional[str] = typer.Argument(
        None, help="Source to ingest: a file path, URL, or '-' for stdin"
    ),
    file: Optional[typer.FileText] = typer.Option(
        None, "--file", "-f", help="Legacy file option (preferred: radar ingest [path])"
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
