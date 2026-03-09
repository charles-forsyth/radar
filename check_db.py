import asyncio
from radar.db.engine import async_session
from sqlalchemy import select, desc
from radar.db.models import Signal
from radar.config import settings
from google.cloud.sql.connector import Connector
from radar.db.engine import set_global_connector


async def check_db():
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
                .where(Signal.title.contains("Dynamic"))
                .order_by(desc(Signal.date))
                .limit(10)
            )
            result = await session.execute(stmt)
            signals = result.scalars().all()
            for s in signals:
                print(f"- {s.title} ({s.date})")
                print(f"  Content length: {len(s.content)}")
    finally:
        if connector:
            await connector.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(check_db())
