import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://www.broadcastify.com/listen/ctid/2299", wait_until="networkidle")
        await page.wait_for_selector(".btable", timeout=15000)
        
        # The numbers are missing from innerText. Let's try grabbing the specific cells.
        rows = await page.query_selector_all("tr")
        print(f"Found {len(rows)} rows.")
        for row in rows:
            cells = await row.query_selector_all("td")
            if len(cells) > 3:
                # Usually: [0] Listen button, [1] Feed Name/Desc, [2] Genre, [3] Listeners
                genre = await cells[2].inner_text()
                if "Public Safety" in genre:
                    feed_name = await cells[1].inner_text()
                    listeners = await cells[3].inner_text()
                    print(f"Feed: {feed_name.replace(chr(10), ' - ')} | Listeners: {listeners}")

        await browser.close()

asyncio.run(run())
