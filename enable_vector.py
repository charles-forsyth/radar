import asyncio
from google.cloud.sql.connector import Connector
import sqlalchemy
from sqlalchemy.ext.asyncio import create_async_engine
from radar.config import settings

async def enable_vector():
    loop = asyncio.get_running_loop()
    connector = Connector(loop=loop)
    
    async def get_conn():
        return await connector.connect_async(
            settings.INSTANCE_CONNECTION_NAME,
            "asyncpg",
            user="postgres",
            password=settings.DB_PASSWORD,
            db="radar", # Connect directly to radar db
            enable_iam_auth=False
        )

    engine = create_async_engine("postgresql+asyncpg://", async_creator=get_conn)
    
    async with engine.connect() as conn:
        print("Connected as postgres user.")
        await conn.execute(sqlalchemy.text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.commit()
        print("Extension 'vector' enabled.")

    await connector.close()

if __name__ == "__main__":
    asyncio.run(enable_vector())
