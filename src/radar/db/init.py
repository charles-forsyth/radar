from sqlmodel import SQLModel
from radar.db.engine import engine

# Import all models so metadata knows about them


async def init_db():
    async with engine.begin() as conn:
        # Create all tables in SQLite
        await conn.run_sync(SQLModel.metadata.create_all)
