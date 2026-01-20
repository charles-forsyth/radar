from sqlmodel import SQLModel
from radar.db.engine import engine


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
