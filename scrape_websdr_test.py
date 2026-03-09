import asyncio
import httpx
from bs4 import BeautifulSoup


async def scrape_websdr():
    url = "http://websdr.ewi.utwente.nl/org/"
    print(f"Scraping {url}...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            if response.status_code != 200:
                print(f"Failed to fetch: {response.status_code}")
                return

            soup = BeautifulSoup(response.text, "html.parser")
            # Find all table rows that might contain SDR info
            # WebSDR.org has a massive table of servers
            rows = soup.find_all("tr")
            found = []

            for row in rows:
                links = row.find_all("a")
                if links:
                    link = links[0].get("href")
                    if link and link.startswith("http"):
                        details = row.get_text(separator=" | ", strip=True)
                        found.append(f"URL: {link}\nDetails: {details}")

            if found:
                print("\nFound Regional WebSDRs:")
                for f in found[:5]:  # Show top 5
                    print("-" * 40)
                    print(f)
            else:
                print("No specific regional WebSDRs found in quick scrape.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(scrape_websdr())
