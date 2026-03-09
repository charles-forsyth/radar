import asyncio
import httpx


async def fetch_kiwisdrs():
    url = "https://kiwisdr.com/public/status.json"
    print(f"Fetching {url}...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=15.0)
            if response.status_code != 200:
                print(f"Failed to fetch: {response.status_code}")
                return

            data = response.json()
            nodes = data.get("stats", [])

            found = []
            for node in nodes:
                loc = str(node.get("loc", "")).lower()
                state = str(node.get("state", "")).lower()
                # Search for Tioga, PA, NY, or surrounding locators (FN11, FN12)
                if any(
                    k in loc
                    for k in ["pennsylvania", "new york", "tioga", "fn1", "fn2"]
                ) or any(k in state for k in ["pa", "ny", "pennsylvania", "new york"]):
                    found.append(
                        {
                            "name": node.get("name"),
                            "location": node.get("loc"),
                            "url": f"http://{node.get('host')}:{node.get('port')}",
                            "users": node.get("users"),
                        }
                    )

            if found:
                print(f"\nFound {len(found)} Regional KiwiSDRs:")
                for f in found[:5]:  # Show top 5
                    print(
                        f"- {f['name']} ({f['location']}): {f['url']} [Users: {f['users']}]"
                    )
            else:
                print("No specific regional KiwiSDRs found.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(fetch_kiwisdrs())
