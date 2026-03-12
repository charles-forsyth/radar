import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://www.broadcastify.com/listen/ctid/2299", wait_until="networkidle")
        await page.wait_for_selector(".btable", timeout=15000)
        
        extracted = await page.evaluate("""() => {
            const results = [];
            const rows = document.querySelectorAll('.btable tr');
            rows.forEach(row => {
                const cells = row.querySelectorAll('td');
                if (cells.length > 3) {
                    const feedName = cells[1].innerText ? cells[1].innerText.trim().split('\\n').join(' - ') : "";
                    const genre = cells[2].innerText ? cells[2].innerText.trim() : "";
                    const listeners = cells[3].innerText ? cells[3].innerText.trim() : "";
                    
                    if (genre.includes('Public Safety') || parseInt(listeners) >= 0) {
                        results.push(`Feed: ${feedName} | Genre: ${genre} | Listeners: ${listeners}`);
                    }
                }
            });
            return results;
        }""")
        print(f"Extracted: {extracted}")
        await browser.close()

asyncio.run(run())
