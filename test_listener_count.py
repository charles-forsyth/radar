import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://www.broadcastify.com/listen/ctid/2299", wait_until="networkidle")
        await page.wait_for_selector(".btable", timeout=15000)
        
        print("Extracting full table html to inspect structure...")
        table_html = await page.evaluate("() => document.querySelector('.btable').outerHTML")
        
        print("--- TABLE STRUCTURE ---")
        # Just print the first relevant row
        for line in table_html.split('\n'):
            if 'Tioga County Police' in line or 'Public Safety' in line or 'class="c"' in line:
                print(line.strip())
        
        await browser.close()

asyncio.run(run())
