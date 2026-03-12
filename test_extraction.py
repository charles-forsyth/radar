import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        url = "https://www.broadcastify.com/listen/ctid/2299"
        await page.goto(url, wait_until="networkidle")
        await asyncio.sleep(4)
        
        rows = await page.query_selector_all("tr")
        print(f"Found {len(rows)} rows.")
        for row in rows:
            text = await row.inner_text()
            if "Public Safety" in text:
                print(f"MATCH: {text}")
        
        await browser.close()

asyncio.run(run())
