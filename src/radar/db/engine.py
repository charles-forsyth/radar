from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from google.cloud.sql.connector import Connector
from radar.config import settings


async def get_db_connection():
    """
    Async creator function for SQLAlchemy.
    Initializes the connector if needed and returns a connection.
    """
    # Instantiate Connector locally to ensure it uses the current event loop
    connector = Connector()

    return await connector.connect_async(
        settings.INSTANCE_CONNECTION_NAME,
        "asyncpg",
        user=settings.DB_USER,
        db=settings.DB_NAME,
        enable_iam_auth=True,
    )


if settings.INSTANCE_CONNECTION_NAME:
    # Use IAM Auth / Cloud SQL Connector
    engine = create_async_engine(
        "postgresql+asyncpg://",
        async_creator=get_db_connection,
        echo=False,
    )
else:
    # Use standard DB_URL
    engine = create_async_engine(settings.DB_URL, echo=False)

async_session = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)
