from sqlmodel import SQLModel, text
from radar.db.engine import engine
# Import all models so metadata knows about them
from radar.db.models import (
    Signal,
    Trend,
    Entity,
    Connection,
    ChatSession,
    ChatMessage,
    Watchpoint,
    Alert,
    TacticalAlert,
    Telemetry,
    RiverLevel,
    RFPeak,
    SoftwareInventory,
    Statistic,
)


async def init_db():
    async with engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(SQLModel.metadata.create_all)
