import asyncio
from playwright.async_api import async_playwright


async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(
            "https://www.broadcastify.com/listen/ctid/2299",
            wait_until="domcontentloaded",
        )
        await asyncio.sleep(5)
        html = await page.content()
        with open("bcf_dump.html", "w") as f:
            f.write(html)
        await browser.close()


asyncio.run(run())
