from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from google.cloud.sql.connector import Connector
from radar.config import settings


# Global connector instance injected at runtime
_global_connector: Connector | None = None


def set_global_connector(connector: Connector):
    """Set the global connector instance for the session."""
    global _global_connector
    _global_connector = connector


async def get_db_connection():
    """
    Async creator function for SQLAlchemy.
    Uses the globally injected connector to establish a connection.
    """
    if not _global_connector:
        raise RuntimeError(
            "Cloud SQL Connector not initialized. Call set_global_connector() first."
        )

    return await _global_connector.connect_async(
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
