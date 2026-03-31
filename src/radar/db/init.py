from sqlmodel import SQLModel
from radar.db.engine import engine


async def init_db():
    from radar.config import settings
    import os

    # Extract absolute path for verbose feedback
    db_path = settings.DB_URL.replace("sqlite+aiosqlite:///", "")
    abs_path = os.path.abspath(db_path)

    print(f"\n[VERBOSE] TARGETING DATABASE: {abs_path}")

    async with engine.begin() as conn:
        print("[VERBOSE] EXECUTING TABLE SCHEMA CREATION...")
        await conn.run_sync(SQLModel.metadata.create_all)

    print("[VERBOSE] SCHEMA VERIFICATION COMPLETE.")
    print(
        "[VERBOSE] TABLES INITIALIZED: signal, telemetry, riverlevel, rfpeak, softwareinventory, statistic, chatsession, chatmessage\n"
    )
