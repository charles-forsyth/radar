import asyncio
from radar.db.engine import async_session
from sqlalchemy import select, delete
from radar.db.models import Entity
from radar.core.ingest import IntelligenceAgent
from radar.config import settings
from google.cloud.sql.connector import Connector
from radar.db.engine import set_global_connector
import uuid


async def test_merge():
    intel = IntelligenceAgent()
    connector = None
    if settings.INSTANCE_CONNECTION_NAME:
        loop = asyncio.get_running_loop()
        connector = Connector(loop=loop)
        set_global_connector(connector)
        await connector.__aenter__()

    try:
        async with async_session() as session:
            print("Fetching two entities to test merge...")
            # Pick two entities that are likely related
            result = await session.execute(select(Entity).limit(2))
            to_merge = result.scalars().all()

            if len(to_merge) < 2:
                print("Not enough entities to test merge.")
                return

            items_data = [
                {
                    "id": str(e.id),
                    "name": e.name,
                    "description": e.details.get("description", ""),
                }
                for e in to_merge
            ]

            print(f"Asking Gemini to merge: {[e.name for e in to_merge]}")
            optimization = await intel.optimize_knowledge(items_data)

            unified_name = optimization["unified_name"]
            unified_desc = optimization["unified_description"]
            merged_ids = [uuid.UUID(mid) for mid in optimization["merged_ids"]]

            print(f"Gemini suggests unified name: {unified_name}")

            # Use the first one as master
            master_entity = to_merge[0]
            master_entity.name = unified_name
            master_entity.details["description"] = unified_desc
            session.add(master_entity)

            for old_id in merged_ids:
                if old_id == master_entity.id:
                    continue

                # Remap and Delete
                print(f"Merging {old_id} into {master_entity.id}")
                await session.execute(delete(Entity).where(Entity.id == old_id))

            await session.commit()
            print("Successfully merged and committed.")

    finally:
        if connector:
            await connector.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(test_merge())
