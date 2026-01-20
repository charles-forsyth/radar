from sqlalchemy.ext.asyncio import create_async_engine
from radar.config import settings

engine = create_async_engine(settings.DB_URL, echo=False)
