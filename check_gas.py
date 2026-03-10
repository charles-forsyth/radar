import asyncio
from radar.db.engine import async_session
from sqlalchemy import select
from radar.db.models import Signal
from radar.config import settings
from google.cloud.sql.connector import Connector
from radar.db.engine import set_global_connector


async def check_gas():
    connector = None
    if settings.INSTANCE_CONNECTION_NAME:
        loop = asyncio.get_running_loop()
        connector = Connector(loop=loop)
        set_global_connector(connector)
        await connector.__aenter__()

    try:
        async with async_session() as session:
            stmt = select(Signal).where(Signal.content.contains("Kwik Fill")).limit(1)
            result = await session.execute(stmt)
            sig = result.scalar_one_or_none()
            if sig:
                lines = sig.content.split("\n")
                for line in lines:
                    if "$" in line and "mi" in line:
                        print(line)
            else:
                print("No gas data found.")
    finally:
        if connector:
            await connector.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(check_gas())
