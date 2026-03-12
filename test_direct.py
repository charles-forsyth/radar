import asyncio
import logging
from radar.core.ingest import BrowserIngestAgent

logging.basicConfig(level=logging.DEBUG)

async def main():
    agent = BrowserIngestAgent()
    url = "https://www.broadcastify.com/listen/ctid/2299"
    print(f"Testing direct extraction for {url}")
    result = await agent.extract(url, "Extract listener counts")
    print("\n--- EXTRACTED RESULT ---")
    print(result)

asyncio.run(main())
