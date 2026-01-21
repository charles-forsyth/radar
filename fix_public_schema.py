import asyncio
from google.cloud.sql.connector import Connector
import sqlalchemy
from sqlalchemy.ext.asyncio import create_async_engine
from radar.config import settings

async def fix_schema_perms():
    loop = asyncio.get_running_loop()
    connector = Connector(loop=loop)
    
    async def get_conn():
        return await connector.connect_async(
            settings.INSTANCE_CONNECTION_NAME,
            "asyncpg",
            user="postgres",
            password=settings.DB_PASSWORD,
            db="radar", 
            enable_iam_auth=False
        )

    engine = create_async_engine("postgresql+asyncpg://", async_creator=get_conn)
    
    async with engine.connect() as conn:
        print("Connected as postgres user.")
        await conn.execute(sqlalchemy.text('GRANT ALL ON SCHEMA public TO "forsythc@ucr.edu";'))
        await conn.execute(sqlalchemy.text('GRANT ALL ON SCHEMA public TO public;'))
        await conn.commit()
        print("Schema permissions granted.")

    await connector.close()

if __name__ == "__main__":
    asyncio.run(fix_schema_perms())
