import asyncio
from radar.core.ingest import IntelligenceAgent

async def main():
    agent = IntelligenceAgent()
    query = "what listener count for tioga?"
    print(f"Searching for: {query}")
    results = await agent.search_signals(query, limit=5)
    for r in results:
        print(f"Match: {r.title}")

asyncio.run(main())
