from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from radar.config import settings

# Use standard local DB_URL exclusively
engine = create_async_engine(settings.DB_URL, echo=False)

async_session = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)
