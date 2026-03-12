import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        # Testing with Tioga County, PA
        url = "https://www.broadcastify.com/listen/ctid/2299"
        await page.goto(url, wait_until="networkidle")
        await asyncio.sleep(2)
        
        # Try to find the listener counts in the table
        rows = await page.query_selector_all("tr")
        print(f"Found {len(rows)} table rows.")
        for row in rows:
            text = await row.inner_text()
            if "Listeners" in text or any(char.isdigit() for char in text):
                # Look for rows that likely contain feed data
                # Usually: [Status Icon] [Feed Name] [Genre] [Listeners] [Player Links]
                print(f"Row: {text.replace('\n', ' | ')}")
        
        await browser.close()

asyncio.run(run())
