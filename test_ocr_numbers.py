import asyncio
from playwright.async_api import async_playwright
import os

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
        await page.goto("https://deflock.org/map#map=10/42.052352/-76.600113", wait_until="networkidle", timeout=20000)
        await asyncio.sleep(3)
        
        # Take a look at the HTML to see what the button actually is
        html = await page.content()
        with open("map_dump.html", "w") as f:
            f.write(html)
            
        await browser.close()

asyncio.run(main())
