import asyncio
from radar.db.engine import engine, set_global_connector
from sqlmodel import text
from radar.config import settings
from google.cloud.sql.connector import Connector


async def migrate_db():
    connector = None
    if settings.INSTANCE_CONNECTION_NAME:
        loop = asyncio.get_running_loop()
        connector = Connector(loop=loop)
        set_global_connector(connector)
        await connector.__aenter__()

    try:
        async with engine.begin() as conn:
            print("Applying Temporal Awareness DB Migration...")
            try:
                await conn.execute(
                    text(
                        "ALTER TABLE entity ADD COLUMN first_seen TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP"
                    )
                )
                await conn.execute(
                    text(
                        "ALTER TABLE entity ADD COLUMN last_seen TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP"
                    )
                )
                print("Entity table updated.")
            except Exception as e:
                print(f"Entity migration info: {e}")

            try:
                await conn.execute(
                    text(
                        "ALTER TABLE trend ADD COLUMN first_seen TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP"
                    )
                )
                await conn.execute(
                    text(
                        "ALTER TABLE trend ADD COLUMN last_seen TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP"
                    )
                )
                print("Trend table updated.")
            except Exception as e:
                print(f"Trend migration info: {e}")

            try:
                await conn.execute(
                    text(
                        "ALTER TABLE connection ADD COLUMN first_seen TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP"
                    )
                )
                await conn.execute(
                    text(
                        "ALTER TABLE connection ADD COLUMN last_seen TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP"
                    )
                )
                print("Connection table updated.")
            except Exception as e:
                print(f"Connection migration info: {e}")

            print("Migration process complete.")
    finally:
        if connector:
            await connector.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(migrate_db())
