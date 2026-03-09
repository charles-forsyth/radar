import asyncio
from radar.db.engine import async_session
from sqlalchemy import select, desc
from radar.db.models import Signal
from radar.config import settings
from google.cloud.sql.connector import Connector
from radar.db.engine import set_global_connector


async def fetch_dynamic():
    connector = None
    if settings.INSTANCE_CONNECTION_NAME:
        loop = asyncio.get_running_loop()
        connector = Connector(loop=loop)
        set_global_connector(connector)
        await connector.__aenter__()

    try:
        async with async_session() as session:
            stmt = (
                select(Signal)
                .where(Signal.title.contains("Dynamic Web Extraction"))
                .order_by(desc(Signal.date))
                .limit(3)
            )
            result = await session.execute(stmt)
            signals = result.scalars().all()
            for s in signals:
                print(f"\n{'=' * 50}")
                print(f"TITLE: {s.title}")
                print(f"DATE: {s.date}")
                print(f"CONTENT PREVIEW:\n{s.content[:300]}...")
                print(f"{'=' * 50}")
    finally:
        if connector:
            await connector.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(fetch_dynamic())
