import json
import logging
import subprocess
from datetime import datetime
from typing import List, Tuple, Optional
import httpx
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from radar.db.models import Signal
from radar.core.models import KnowledgeGraphExtraction

logger = logging.getLogger(__name__)


class IntelligenceAgent:
    def __init__(self):
        # Local NLP tools paths
        self.embed_bin = "src/radar/tools/radar_embed"
        self.extract_bin = "src/radar/tools/radar_extract"
        self.summarize_bin = "src/radar/tools/radar_summarize"
        self.fetch_bin = "src/radar/tools/radar_fetch"

    def _run_tool(self, tool: str, text: str) -> str:
        try:
            result = subprocess.run(
                [tool], input=text, text=True, capture_output=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running {tool}: {e.stderr}")
            return ""

    def _fetch_url(self, url: str) -> str:
        try:
            # Check if it's a PDF
            is_pdf = url.lower().endswith(".pdf")
            
            # Run fetch tool
            result = subprocess.run(
                [self.fetch_bin, url], capture_output=True, check=True
            )
            
            if is_pdf:
                # Save binary to temp file and run pdftotext
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(result.stdout)
                    tmp_path = tmp.name
                
                try:
                    # pdftotext -layout <file> -
                    pdf_result = subprocess.run(
                        ["pdftotext", "-layout", tmp_path, "-"],
                        capture_output=True, text=True, check=True
                    )
                    return pdf_result.stdout.strip()
                except Exception as pdf_e:
                    logger.error(f"PDF extraction failed for {url}: {pdf_e}")
                    return ""
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            
            return result.stdout.decode('utf-8', errors='ignore').strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Error fetching {url}: {e.stderr.decode('utf-8', errors='ignore')}")
            return ""

    async def get_embedding(self, text: str) -> List[float]:
        """Generate a vector embedding using local C tool."""
        output = self._run_tool(self.embed_bin, text)
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return [0.0] * 3072  # Match the updated dimensions

    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        results = []
        for text in texts:
            results.append(await self.get_embedding(text))
        return results

    async def parse(self, text: str) -> Tuple[Signal, KnowledgeGraphExtraction]:
        """Extract structured entities using local C tool."""
        output = self._run_tool(self.extract_bin, text)
        try:
            kg_data = json.loads(output)
            kg = KnowledgeGraphExtraction(**kg_data)
        except Exception as e:
            logger.error(f"Parse error: {e}")
            kg = KnowledgeGraphExtraction(entities=[], connections=[], trends=[])

        signal = Signal(
            title=text.split("\n")[0][:255],
            content=text,
            source="local_nlp",
            date=datetime.now(),
        )

        return signal, kg

    async def answer_question(
        self, question: str, context_signals: List[Signal]
    ) -> str:
        # This is a bit hacky since we don't have the session here,
        # but the passed context_signals should ideally contain the hits.
        # If they don't, it means the vector search failed.

        context_text = "\n\n".join(
            [f"--- Signal: {s.title} ---\n{s.content}" for s in context_signals]
        )
        return self._run_tool(
            self.summarize_bin, f"Question: {question}\nContext: {context_text}"
        )

    async def optimize_knowledge(self, items: List[dict]) -> dict:
        return {
            "unified_name": "Local Merged Entity",
            "unified_description": "Merged by local tool",
            "merged_ids": [],
        }

    async def generate_report(self, topic: str, context: dict) -> str:
        summary = self._run_tool(self.summarize_bin, str(context))
        return f"# Report on {topic}\n\n{summary}"

    async def generate_briefing(self, context: dict) -> str:
        return self._run_tool(self.summarize_bin, str(context))

    async def check_watchpoints(
        self, signal_content: str, watchpoints: List[dict]
    ) -> List[dict]:
        matches = []
        content_lower = signal_content.lower()
        for wp in watchpoints:
            if str(wp.get("id", "")).lower() in content_lower:
                matches.append(
                    {"watchpoint_id": wp.get("id"), "reason": "Local keyword match"}
                )
        return matches

    async def detect_anomalies(
        self, current_sitrep: str, baseline_context: str
    ) -> List[dict]:
        if "critical" in current_sitrep.lower():
            return [
                {
                    "domain": "GENERAL",
                    "severity": "CRITICAL",
                    "message": "Local anomaly detected",
                }
            ]
        return []

    async def chat(
        self, question: str, context_signals: List[Signal], history: List[dict] = []
    ) -> str:
        return await self.answer_question(question, context_signals)


class DeepResearchAgent:
    def __init__(self):
        self.intel = IntelligenceAgent()
        # Fallback sources if search fails
        self.sources = [
            "https://www.hpcwire.com/",
            "https://www.rtl-sdr.com/",
            "https://www.hackaday.com/",
            "https://thehackernews.com/",
            "https://www.phoronix.com/",
            "https://www.radioreference.com/",
        ]

    async def research(self, topic: str) -> str:
        logger.info(f"Performing local autonomous research on: {topic}")
        combined_text = ""

        # 1. Use local DuckDuckGo scraper library to find relevant URLs
        urls_to_scrape = []
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(topic, max_results=3))
                for r in results:
                    if "href" in r:
                        urls_to_scrape.append(r["href"])
        except Exception as e:
            logger.error(f"DDGS search failed for {topic}: {e}")

        # Fallback to static sources if search fails
        if not urls_to_scrape:
            urls_to_scrape = self.sources

        # 2. Fetch the content of the discovered URLs
        topic_words = [w.lower() for w in topic.split() if len(w) > 3]

        from playwright.async_api import async_playwright
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                # Create a context that masks our automated nature
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    viewport={"width": 1920, "height": 1080},
                    java_script_enabled=True,
                )
                page = await context.new_page()
                
                for url in urls_to_scrape:
                    try:
                        # Skip PDFs for Playwright as it will just download them; use our existing C fetcher for PDFs
                        if url.lower().endswith(".pdf"):
                            content = self.intel._fetch_url(url)
                        else:
                            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                            
                            # Give modern JS frameworks a second to paint their DOM
                            import asyncio
                            await asyncio.sleep(2)
                            
                            # Extract visible text directly using Playwright's evaluator to skip HTML tags entirely
                            content = await page.evaluate("() => document.body.innerText")

                        content_lower = content.lower()

                        # Verify relevance before appending
                        hit = False
                        for word in topic_words:
                            if word in content_lower:
                                hit = True
                                break

                        if hit or not topic_words:
                            # Truncate content to avoid overwhelming the C summarizer buffer
                            combined_text += f"\n--- Source: {url} ---\n{content[:4000]}"
                    except Exception as e:
                        logger.error(f"Failed to fetch {url} with Playwright: {e}")
                
                await browser.close()
        except Exception as browser_err:
            logger.error(f"Playwright initialization failed: {browser_err}")
            return f"Local research failed to initialize browser for: {topic}"

        if not combined_text:
            return f"Local research found no direct hits on {topic} in monitored high-value domains."

        summary = self.intel._run_tool(
            self.intel.summarize_bin, f"Question: {topic}\n{combined_text}"
        )
        return f"Autonomous Research Report for: {topic}\n\n{summary}"


class USGSRiverGauge:
    def __init__(
        self, site_codes: List[str] = ["01548500", "01531500", "01531000", "01518700"]
    ):
        self.site_codes = site_codes

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_levels(self) -> str:
        sites_str = ",".join(self.site_codes)
        url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={sites_str}&parameterCd=00060,00065&siteStatus=all"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                if response.status_code != 200:
                    return f"USGS API Error: {response.status_code}"
                data = response.json()
                results = []
                if "value" in data and "timeSeries" in data["value"]:
                    for ts in data["value"]["timeSeries"]:
                        site_name = ts["sourceInfo"]["siteName"]
                        param_name = ts["variable"]["variableName"]
                        if not ts["values"][0]["value"]:
                            continue
                        latest_val = ts["values"][0]["value"][0]["value"]
                        unit = "ft" if "height" in param_name.lower() else "cfs"
                        results.append(f"- {site_name}: {latest_val} {unit}")
                if not results:
                    return "No active gauge data found."
                return "\n".join(sorted(list(set(results))))
        except Exception as e:
            return f"USGS Error: {str(e)}"


class NWSAlerts:
    def __init__(self, lat: float = 41.8, lon: float = -77.1):
        self.lat = lat
        self.lon = lon
        self.headers = {"User-Agent": "RadarTacticalHUD/1.0"}

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_alerts(self) -> str:
        url = f"https://api.weather.gov/alerts/active?point={self.lat},{self.lon}"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=self.headers, timeout=10.0)
                if response.status_code != 200:
                    return f"NWS API Error: {response.status_code}"
                data = response.json()
                features = data.get("features", [])
                if not features:
                    return "No active severe weather alerts for this sector."
                alerts = []
                for f in features:
                    props = f.get("properties", {})
                    headline = props.get("headline", "Unknown Alert")
                    alerts.append(f"⚠️ {headline}")
                return "\n".join(alerts)
        except Exception as e:
            return f"NWS Error: {str(e)}"


class CISAFeed:
    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_latest_vulns(self) -> str:
        url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                if response.status_code != 200:
                    return f"CISA API Error: {response.status_code}"
                data = response.json()
                vulns = data.get("vulnerabilities", [])
                if not vulns:
                    return "No CISA vulnerability data found."
                recent = sorted(
                    vulns, key=lambda x: x.get("dateAdded", ""), reverse=True
                )[:3]
                lines = []
                for v in recent:
                    lines.append(
                        f"- {v.get('cveID')}: {v.get('vulnerabilityName')} (Added: {v.get('dateAdded')})"
                    )
                return "\n".join(lines)
        except Exception as e:
            return f"CISA Error: {str(e)}"


class NewsWire:
    def __init__(self):
        self.feeds = [
            "https://www.hackaday.com/blog/feed/",
            "https://www.rtl-sdr.com/feed/",
            "https://www.hpcwire.com/feed/",
            "https://feeds.feedburner.com/TheHackersNews",
        ]

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_headlines(self) -> str:
        all_headlines = []
        async with httpx.AsyncClient() as client:
            for url in self.feeds:
                try:
                    response = await client.get(url, timeout=10.0)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, "xml")
                        items = soup.find_all("item")[:3]
                        for item in items:
                            if item.title:
                                title = item.title.text.strip()
                                all_headlines.append(f"- {title}")
                except Exception:
                    continue
        if not all_headlines:
            return "No breaking news found in technical wires."
        return "\n".join(all_headlines[:10])


class RSSIngestAgent:
    def __init__(self):
        self.wire = NewsWire()
        self.intel = IntelligenceAgent()

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def sync_news(self) -> List[Tuple[Signal, KnowledgeGraphExtraction]]:
        results = []
        async with httpx.AsyncClient() as client:
            for url in self.wire.feeds:
                try:
                    response = await client.get(url, timeout=10.0)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, "xml")
                        items = soup.find_all("item")[:2]
                        for item in items:
                            if item.title and item.link:
                                title = item.title.text.strip()
                                link = item.link.text.strip()
                            else:
                                continue
                            article_resp = await client.get(link, timeout=15.0)
                            if article_resp.status_code == 200:
                                art_soup = BeautifulSoup(
                                    article_resp.text, "html.parser"
                                )
                                text = art_soup.get_text(separator=" ", strip=True)
                                final_text = f"Title: {title}\nSource: {url}\nLink: {link}\n\n{text[:10000]}"
                                signal, kg = await self.intel.parse(final_text)
                                results.append((signal, kg))
                except Exception:
                    continue
        return results


class SectorScanner:
    def __init__(self, location: str = "Tioga County, PA"):
        self.location = location

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_metar(self) -> str:
        url = "https://aviationweather.gov/api/data/metar?ids=KELM"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                if response.status_code == 200:
                    return response.text.strip()
                return f"Weather Error: {response.status_code}"
        except Exception as e:
            return f"Weather Error: {str(e)}"

    async def get_atmos_weather(self) -> str:
        try:
            result = subprocess.run(
                ["/home/chuck/.local/bin/atmos", self.location],
                capture_output=True,
                text=True,
                timeout=15.0,
            )
            import re

            clean_text = re.sub(r"\x1b\[[0-9;]*m", "", result.stdout)
            return clean_text.strip()
        except Exception as e:
            return f"Atmos Error: {str(e)}"

    async def get_snapshot_text(self) -> str:
        metar = await self.get_metar()
        atmos = await self.get_atmos_weather()
        sat_scanner = SatelliteScanner()
        sat_passes = await sat_scanner.get_next_passes()
        grid = GridScanner()
        grid_status = await grid.get_summary()
        return f"### SECTOR OPS SITREP\n\n#### Professional Weather (Atmos)\n{atmos}\n\n#### Aviation Weather (KELM)\n{metar}\n\n#### Grid Stability\n{grid_status}\n\n#### Orbital\n{sat_passes}"


class GridScanner:
    def __init__(self):
        self.browser = BrowserIngestAgent()
        self.penelec_url = "https://outages.firstenergycorp.com/pa.html"
        self.tri_county_url = "https://outagemap.tri-countyrec.com/"
        self.nyseg_url = "https://outagemap.nyseg.com/"
        self.ppl_url = "https://omap.pplweb.com/OMAP"

    async def get_summary(self) -> str:
        import asyncio

        penelec_task = self.browser.extract(self.penelec_url, "Extract outages Penelec")
        tri_county_task = self.browser.extract(
            self.tri_county_url, "Extract outages Tri-County"
        )
        nyseg_task = self.browser.extract(self.nyseg_url, "Extract outages NYSEG")
        ppl_task = self.browser.extract(self.ppl_url, "Extract outages PPL")
        results = await asyncio.gather(
            penelec_task, tri_county_task, nyseg_task, ppl_task
        )
        return (
            f"- **Penelec (PA):** {results[0]}\n"
            f"- **Tri-County REC (PA):** {results[1]}\n"
            f"- **NYSEG (NY):** {results[2]}\n"
            f"- **PPL (PA):** {results[3]}"
        )


class SatelliteScanner:
    def __init__(self, lat: float = 41.8, lon: float = -77.1):
        self.lat = lat
        self.lon = lon
        self.targets = {
            "ISS": "25544",
            "NOAA 15": "25338",
            "NOAA 18": "28654",
            "NOAA 19": "33591",
            "METEOR M2-3": "57166",
        }

    async def get_next_passes(self) -> str:
        from skyfield.api import Topos, load
        from datetime import datetime, timedelta

        try:
            stations = load.tle_file("https://celestrak.org/NORAD/elements/weather.txt")
            stations_iss = load.tle_file(
                "https://celestrak.org/NORAD/elements/stations.txt"
            )
            ts = load.timescale()
            t0 = ts.now()
            t1 = ts.from_datetime(datetime.now() + timedelta(hours=12))
            location = Topos(latitude_degrees=self.lat, longitude_degrees=self.lon)
            results = []
            for name, norad_id in self.targets.items():
                sat = next(
                    (
                        s
                        for s in (stations + stations_iss)
                        if norad_id in s.name or name in s.name
                    ),
                    None,
                )
                if sat:
                    t, events = sat.find_events(location, t0, t1, altitude_degrees=10.0)
                    if len(events) > 0:
                        for ti, event in zip(t, events):
                            if event == 1:
                                local_time = ti.utc_datetime() - timedelta(hours=4)
                                results.append(
                                    f"- **{name}:** {local_time.strftime('%H:%M')} (Alt: {sat.at(ti).subpoint().elevation.km:.0f}km)"
                                )
                                break
            if not results:
                return "No high-priority satellite passes detected."
            return "\n".join(results)
        except Exception as e:
            return f"Orbital Calculation Error: {str(e)}"


class TacticalAgent:
    def __init__(self):
        self.adsb = ADSBScanner()
        self.aprs = APRSStreamer()
        self.sector = SectorScanner()
        self.usgs = USGSRiverGauge()
        self.nws = NWSAlerts()
        self.cisa = CISAFeed()

    async def generate_full_sitrep(self, previous_sitrep: Optional[str] = None) -> str:
        from datetime import datetime
        import asyncio

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        (
            adsb_data,
            weather_data,
            aprs_data,
            usgs_data,
            nws_data,
            cisa_data,
        ) = await asyncio.gather(
            self.adsb.get_snapshot_text(),
            self.sector.get_atmos_weather(),
            self.aprs.get_snapshot_text(),
            self.usgs.get_levels(),
            self.nws.get_alerts(),
            self.cisa.get_latest_vulns(),
        )

        delta_analysis = (
            "\n## LOCAL TACTICAL DELTAS\n[Local anomaly logic placeholder]\n"
            if previous_sitrep
            else ""
        )

        sitrep = f"""Title: Master Tactical SITREP - {timestamp}
Source: Integrated Sensor Array
Location: Tioga County Sector (41.8N, 77.1W)
{delta_analysis}
## ENVIRONMENTAL & SUSTAINMENT DATA
- **Aviation Weather (Atmos):** 
{weather_data}

- **Severe Weather Alerts (NWS):** 
{nws_data}

- **Hydrology (USGS River Gauges):**
{usgs_data}

## SIGNAL INTELLIGENCE (SIGINT)
{adsb_data}

{aprs_data}

## CYBER INTELLIGENCE (CISA KEV)
- **Recently Exploited Vulnerabilities:**
{cisa_data}
"""
        return sitrep


class ADSBScanner:
    def __init__(self, host: str = "192.168.1.246", user: str = "pi"):
        self.host = host
        self.user = user

    async def get_live_data(self) -> dict:
        import json
        import asyncio

        cmd = [
            "ssh",
            f"{self.user}@{self.host}",
            "cat /home/pi/adsb_data/aircraft.json",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                return {"error": f"SSH Error: {stderr.decode()}"}
            return json.loads(stdout.decode())
        except Exception as e:
            return {"error": str(e)}

    async def get_snapshot_text(self) -> str:
        data = await self.get_live_data()
        if "error" in data:
            return f"ADS-B Sensor Error: {data['error']}"
        aircraft = data.get("aircraft", [])
        if not aircraft:
            return "No aircraft currently detected overhead."
        lines = ["### ADS-B SITREP - Aircraft Overhead"]
        for ac in aircraft:
            flight = ac.get("flight", "Unknown").strip()
            icao = ac.get("hex", "N/A")
            alt = ac.get("alt_baro", 0)
            gs = ac.get("gs", 0)
            lines.append(
                f"- Flight {flight} (ICAO: {icao}) at {alt:,} ft, Ground Speed: {gs:.0f} kt"
            )
        return "\n".join(lines)


class APRSStreamer:
    def __init__(
        self,
        host: str = "noam.aprs2.net",
        port: int = 14580,
        callsign: str = "NOCALL",
        filter_str: str = "r/41.8/-77.1/100",
    ):
        self.host = host
        self.port = port
        self.callsign = callsign
        self.filter_str = filter_str
        self.packets: List[str] = []
        self.max_packets = 50

    async def get_snapshot_text(self) -> str:
        import asyncio

        if self.packets:
            lines = ["### APRS SITREP - Local Radio Traffic"]
            for p in self.packets[-15:]:
                lines.append(f"- {p}")
            return "\n".join(lines)
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), timeout=5.0
            )
            login_str = f"user {self.callsign} pass -1 vers RadarIntelligence 0.1 filter {self.filter_str}\n"
            writer.write(login_str.encode())
            await writer.drain()
            lines = ["### APRS SITREP - Local Radio Traffic"]
            packets_found = False
            start_time = asyncio.get_event_loop().time()
            while (asyncio.get_event_loop().time() - start_time) < 3.0:
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=1.0)
                    if not line:
                        break
                    packet_text = line.decode().strip()
                    if packet_text and not packet_text.startswith("#"):
                        lines.append(f"- {packet_text}")
                        packets_found = True
                except asyncio.TimeoutError:
                    continue
            writer.close()
            await writer.wait_closed()
            if packets_found:
                return "\n".join(lines)
            return "No local APRS traffic detected."
        except Exception as e:
            return f"APRS Sensor Error: {e}"

    async def start_stream(self):
        import asyncio

        reader, writer = await asyncio.open_connection(self.host, self.port)
        login_str = f"user {self.callsign} pass -1 vers RadarIntelligence 0.1 filter {self.filter_str}\n"
        writer.write(login_str.encode())
        await writer.drain()
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                packet_text = line.decode().strip()
                if packet_text.startswith("#"):
                    continue
                self.packets.append(packet_text)
                if len(self.packets) > self.max_packets:
                    self.packets.pop(0)
        finally:
            writer.close()
            await writer.wait_closed()


class BrowserIngestAgent:
    def __init__(self):
        self.intel = IntelligenceAgent()

    async def extract(self, url: str, instructions: str) -> str:
        """Fetch dynamic JS content using local Playwright and summarize. Fallback to OCR if requested."""
        from playwright.async_api import async_playwright
        import asyncio
        import os
        import tempfile

        content = ""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    viewport={"width": 1920, "height": 1080},
                    java_script_enabled=True,
                )
                page = await context.new_page()
                
                logger.info(f"Playwright navigating to dynamic target: {url}")
                await page.goto(url, wait_until="networkidle", timeout=20000)
                
                # Give complex SPA frameworks time to render API data
                await asyncio.sleep(4)
                
                # Extract visible text directly using Playwright
                content = await page.evaluate("() => document.body.innerText")
                
                # If content is suspiciously short or instructions mention "map", take a screenshot and OCR it
                if len(content) < 1000 or "map" in url.lower() or "ocr" in instructions.lower():
                    logger.info(f"Triggering local OCR for {url}")
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        screenshot_path = tmp.name
                    
                    await page.screenshot(path=screenshot_path, full_page=True)
                    
                    try:
                        import pytesseract
                        from PIL import Image
                        ocr_text = pytesseract.image_to_string(Image.open(screenshot_path))
                        content += f"\n--- OCR EXTRACTION ---\n{ocr_text}"
                    except Exception as ocr_err:
                        logger.error(f"Local OCR failed: {ocr_err}")
                    finally:
                        if os.path.exists(screenshot_path):
                            os.remove(screenshot_path)
                
                await browser.close()
        except Exception as e:
            logger.error(f"Playwright dynamic extraction failed for {url}: {e}")
            # Fallback to static C fetcher if Playwright crashes
            content = self.intel._fetch_url(url)

        return self.intel._run_tool(
            self.intel.summarize_bin,
            f"Question: {instructions}\nTarget: {url}\n\n{content[:15000]}",
        )


class WebIngestAgent:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.intel = IntelligenceAgent()

    async def fetch(self, url: str) -> str:
        return self.intel._fetch_url(url)

    async def ingest(self, url: str) -> Tuple[Signal, KnowledgeGraphExtraction]:
        text = self.fetch(url)
        full_text = f"Source URL: {url}\nDate: {datetime.now()}\n\n{text}"
        signal, kg = await self.intel.parse(full_text)
        signal.url = url
        return signal, kg


class TextIngestAgent:
    def __init__(self):
        self.intel = IntelligenceAgent()

    async def ingest(self, text: str) -> Tuple[Signal, KnowledgeGraphExtraction]:
        logger.info("Ingesting raw text from stdin")
        signal, kg = await self.intel.parse(text)
        logger.info(f"Parsed signal: {signal.title}")
        return signal, kg
