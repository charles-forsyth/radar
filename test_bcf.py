import asyncio
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://www.broadcastify.com/listen/ctid/2299", wait_until="networkidle")
        
        extracted_feeds = await page.evaluate("""() => {
            const results = [];
            const rows = document.querySelectorAll('.btable tr');
            rows.forEach(row => {
                const cells = row.querySelectorAll('td');
                if (cells.length > 3) {
                    const feedName = cells[1].innerText.trim().replace(/\n/g, ' - ');
                    const genre = cells[2].innerText.trim();
                    const listeners = cells[3].innerText.trim();
                    
                    if (genre.includes('Public Safety') || parseInt(listeners) >= 0) {
                        results.push(`Feed: ${feedName} | Genre: ${genre} | Listeners: ${listeners}`);
                    }
                }
            });
            return results;
        }""")
        
        print(f"Extracted {len(extracted_feeds)} feeds:")
        for f in extracted_feeds:
            print(f)
            
        await browser.close()

asyncio.run(run())
