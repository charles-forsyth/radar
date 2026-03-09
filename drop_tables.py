import asyncio
from sqlmodel import text
from google.cloud.sql.connector import Connector
from radar.db.engine import engine, set_global_connector
from radar.config import settings


async def drop_db():
    connector = None
    if settings.INSTANCE_CONNECTION_NAME:
        loop = asyncio.get_running_loop()
        connector = Connector(loop=loop)
        set_global_connector(connector)
        await connector.__aenter__()

    try:
        async with engine.begin() as conn:
            print("Dropping tables...")
            await conn.execute(text("DROP TABLE IF EXISTS signal CASCADE"))
            await conn.execute(text("DROP TABLE IF EXISTS trend CASCADE"))
            await conn.execute(text("DROP TABLE IF EXISTS entity CASCADE"))
            await conn.execute(text("DROP TABLE IF EXISTS connection CASCADE"))
            print("Done.")
    finally:
        if connector:
            await connector.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(drop_db())
