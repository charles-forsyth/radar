import httpx
from bs4 import BeautifulSoup
from radar.db.models import Signal
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WebIngestAgent:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (compatible; RadarBot/0.1; +http://github.com/charles-forsyth/radar)"
        }

    async def fetch(self, url: str) -> str:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url, headers=self.headers, follow_redirects=True
                )
                response.raise_for_status()
                return response.text
            except httpx.HTTPError as e:
                logger.error(f"HTTP Error fetching {url}: {e}")
                raise

    def parse(self, html: str, url: str) -> Signal:
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.string if soup.title else "No Title"

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator="\n", strip=True)

        return Signal(
            title=title,
            url=url,
            content=text[:5000],  # Basic truncation
            raw_text=text,
            date=datetime.now(),
            source="web",
        )

    async def ingest(self, url: str) -> Signal:
        logger.info(f"Ingesting {url}")
        html = await self.fetch(url)
        signal = self.parse(html, url)
        logger.info(f"Parsed signal: {signal.title}")
        return signal
