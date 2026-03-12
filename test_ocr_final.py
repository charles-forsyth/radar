import asyncio
from radar.core.ingest import BrowserIngestAgent

async def main():
    agent = BrowserIngestAgent()
    print("Testing dynamic OCR scrape on map for NUMBERS: https://deflock.org/map#map=10/42.052352/-76.600113")
    url = "https://deflock.org/map#map=10/42.052352/-76.600113"
    instruction = "extract numbers of cameras"
    result = await agent.extract(url, instruction)
    print(result)

asyncio.run(main())
