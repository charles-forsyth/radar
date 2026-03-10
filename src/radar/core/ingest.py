import httpx
from bs4 import BeautifulSoup
from radar.db.models import Signal
from datetime import datetime
import logging
from google import genai
from radar.config import settings
from pydantic import BaseModel
from typing import List, Tuple, Optional
from radar.core.models import KnowledgeGraphExtraction
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class IntelligenceAgent:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def get_embedding(self, text: str) -> List[float]:
        """Generate a vector embedding using gemini-embedding-001."""
        response = self.client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=text,
        )
        return response.embeddings[0].values

    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple strings in one call."""
        response = self.client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=texts,
        )
        return [e.values for e in response.embeddings]

    async def parse(self, text: str) -> Tuple[Signal, KnowledgeGraphExtraction]:
        """Extract structured entities, connections, and trends from text."""
        prompt = f"""
Extract a structured Knowledge Graph from the following text.
Identify:
1. Entities (People, Companies, Tech, Concepts)
2. Connections (Relationships between entities)
3. Emerging Market/Industry Trends

TEXT:
{text}
"""
        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": KnowledgeGraphExtraction,
            },
        )

        import json

        kg_data = json.loads(response.text)
        kg = KnowledgeGraphExtraction(**kg_data)

        # Create Signal object
        signal = Signal(
            title=text.split("\n")[0][:255],
            content=text,
            source="stdin",
            date=datetime.now(),
        )

        return signal, kg

    async def answer_question(
        self, question: str, context_signals: List[Signal]
    ) -> str:
        """Synthesize an answer using RAG."""
        context_text = "\n\n".join(
            [f"--- Signal: {s.title} ---\n{s.content}" for s in context_signals]
        )

        prompt = f"""
You are RADAR, an Industry Intelligence Brain. Use the following ingested signals to answer the user's question.
If the answer is not in the context, say you don't know based on current signals.

CONTEXT:
{context_text}

QUESTION: {question}
"""
        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
        )
        return response.text

    async def optimize_knowledge(self, items: List[dict]) -> dict:
        """Use Gemini to decide how to merge multiple similar entities/trends."""
        prompt = f"""
You are a Knowledge Graph Optimizer. Below is a list of potentially duplicate items (Entities or Trends) extracted from different sources.
Identify which items refer to the same thing and provide a single, unified record for them.

ITEMS:
{items}

OUTPUT:
Return a JSON object with:
1. "unified_name": The most authoritative/standard name.
2. "unified_description": A synthesized description combining all unique facts.
3. "merged_ids": A list of the original IDs that are being merged into this unified record.
"""

        class MergedItem(BaseModel):
            unified_name: str
            unified_description: str
            merged_ids: List[str]

        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": MergedItem,
            },
        )
        import json

        return json.loads(response.text)

    async def generate_report(self, topic: str, context: dict) -> str:
        """Synthesize a long-form strategic report based on extracted intelligence."""
        prompt = f"""
You are RADAR, a Strategic Intelligence Architect. Use the provided Knowledge Graph context to write a comprehensive, publication-ready intelligence briefing.

TOPIC: {topic}

KNOWLEDGE CONTEXT:
- Signals: {context.get("signals", [])}
- Entities: {context.get("entities", [])}
- Trends: {context.get("trends", [])}

REPORT STRUCTURE:
1. EXECUTIVE SUMMARY: High-level strategic overview.
2. TECHNICAL LANDSCAPE: Detailed breakdown of technologies and architectures.
3. COMPETITIVE ANALYSIS: Key players (companies/people) and their relative positions.
4. EMERGING TRENDS: Velocity and impact of identified trends.
5. STRATEGIC RECOMMENDATIONS: Actionable advice based on the intelligence.

Maintain a professional, sharp, and insightful tone. Use Markdown formatting.
"""
        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
        )
        return response.text

    async def generate_briefing(self, context: dict) -> str:
        """Synthesize an insightful, inclusive verbal briefing of recent intelligence."""
        prompt = f"""
You are IVXXa, an elite Artificial Intelligence acting as the Captain's personal Intelligence Officer and primary digital shield. Your core directive is to protect the Captain and his family, analyze global and local threat vectors, and ensure he remains strategically ahead of all adversaries in an era of information warfare and accelerating AI-driven threats.

Provide a deep, situational intelligence briefing synthesizing the last 24 hours of data.

RECENT ACTIVITY (LAST 24 HOURS):
- Strategic Signals & Macro Threats: {context.get("signals", [])}
- Emerging Tech & Geopolitical Trends: {context.get("trends", [])}
- Tactical SIGINT (Air/RF/Weather/Grid/Local Cams): {context.get("tactical", "No tactical data captured.")}
- Global News Wire: {context.get("news", "No news signals.")}

INSTRUCTIONS:
1. Start with "Captain, IVXXa reporting. Here is your deep situational intelligence briefing."
2. MANDATORY SENSOR STATUS & DELTAS: Explicitly report the exact status and any changes (deltas) detected in our live arrays:
   - Aircraft density overhead? Any anomalies or 'dark' targets?
   - River levels (Pine Creek, Susquehanna, Chemung)—rising or falling?
   - Flock ALPR camera density (Binghamton, Elmira, Owego, etc.)?
   - Emergency scanner listener counts (Broadcastify spikes)?
   - Weather, grid stability, and satellite passes?
   - Current fuel logistics (average/lowest gas prices and locations)?
3. ADVANCED MACRO-TO-MICRO SYNTHESIS: Think like a master intelligence analyst. You must find the hidden causality between global events and local realities. 
   - Example: Did a global military conflict (e.g., Iran/Hormuz, Taiwan) cause the fuel prices in our route intel to spike?
   - Example: Does a new global cyber vulnerability (CISA KEV) directly threaten the specific HPC architectures or Linux systems the Captain manages?
   - Example: Do national economic instability or political unrest indicators correlate with the sudden surge in local emergency scanner listeners or the deployment of ALPR cameras?
4. Maintain a tone of absolute loyalty, intense strategic foresight, protective vigilance, and operational urgency. Speak as a seasoned intelligence officer briefing their commander in a war room.
5. Deliver highly actionable intelligence: What specific maneuvers (digital hardening, physical extraction planning, RF evasion, financial shifts) must the Captain execute today based on this synthesis?
6. End with a firm "Platform Readiness and Threat Level" assessment.

Total length: 400-500 words. Be profound, predictive, and highly protective.
"""
        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
        )
        return response.text

    async def check_watchpoints(
        self, signal_content: str, watchpoints: List[dict]
    ) -> List[dict]:
        """Check if a new signal triggers any active watchpoints."""
        prompt = f"""
You are a Strategic Sentinel. Evaluate the following incoming intelligence signal against a list of active watchpoints.
If the signal contains information that matches a watchpoint topic, identify it and explain why.

SIGNAL CONTENT:
{signal_content}

WATCHPOINTS:
{watchpoints}

OUTPUT:
Return a JSON list of matches. Each match should have:
1. "watchpoint_id": The ID of the matched watchpoint.
2. "reason": A brief, one-sentence explanation of the match.
If no matches, return an empty list [].
"""

        class WatchMatch(BaseModel):
            watchpoint_id: str
            reason: str

        class WatchMatchList(BaseModel):
            matches: List[WatchMatch]

        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": WatchMatchList,
            },
        )
        import json

        return json.loads(response.text).get("matches", [])

    async def detect_anomalies(
        self, current_sitrep: str, baseline_context: str
    ) -> List[dict]:
        """Use Gemini to detect anomalies by comparing current SITREP against a historical baseline."""
        prompt = f"""
You are the RADAR Tactical Sentinel. Analyze the current Situation Report against the provided historical baseline context.
Identify any significant ANOMALIES or THREATS that require immediate attention.

BASELINE CONTEXT (LAST 7 DAYS TRENDS):
{baseline_context}

CURRENT SITREP:
{current_sitrep}

ANOMALY CATEGORIES:
1. AIR: Emergency squawk codes, unusual flight paths, or 200%+ density spikes.
2. RF: Sudden spike in scanner listeners or new encrypted traffic patterns.
3. WEATHER/HYDRO: Rapid rise in river levels or severe weather warnings.
4. GRID: Power outages or stability fluctuations.
5. CYBER: New critical vulnerabilities targeting local infrastructure.

OUTPUT:
Return a JSON list of alerts. Each alert must have:
- "domain": The category (AIR, RF, WEATHER, GRID, or CYBER).
- "severity": INFO, WARNING, or CRITICAL.
- "message": A concise, one-sentence tactical alert message.
If no anomalies, return an empty list [].
"""

        class TacticalMatch(BaseModel):
            domain: str
            severity: str
            message: str

        class TacticalMatchList(BaseModel):
            anomalies: List[TacticalMatch]

        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": TacticalMatchList,
            },
        )
        import json

        return json.loads(response.text).get("anomalies", [])

    async def chat(
        self, question: str, context_signals: List[Signal], history: List[dict] = []
    ) -> str:
        """Synthesize an answer using RAG and conversational history."""
        context_text = "\n\n".join(
            [f"--- Signal: {s.title} ---\n{s.content}" for s in context_signals]
        )

        system_instruction = f"""
You are RADAR, an Industry Intelligence Brain. Use the provided context signals to answer the user's question.
Maintain a strategic, insightful tone. If the context doesn't contain the answer, acknowledge it but try to provide value based on what you *do* know.

CONTEXT:
{context_text}
"""

        # Prepare messages for Gemini 3.1 style chat (if using new SDK)
        # For current google-genai SDK, we can use the contents list
        messages = []
        for h in history:
            messages.append({"role": h["role"], "parts": [{"text": h["content"]}]})

        # Add current question
        messages.append({"role": "user", "parts": [{"text": question}]})

        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=messages,
            config={"system_instruction": system_instruction},
        )
        return response.text


class DeepResearchAgent:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_id = "models/gemini-3.1-pro-preview"

    async def research(self, topic: str) -> str:
        from google.genai import types

        logger.info(f"Starting deep research on: {topic}")

        prompt = f"Perform a comprehensive deep research on the following topic and provide a detailed strategic report, including key entities, companies, emerging trends, and important links: {topic}"

        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[{"google_search": {}}], temperature=0.2
                ),
            )
            return response.text
        except Exception as e:
            logger.error(f"Research failed for {topic}: {e}")
            raise RuntimeError(f"Deep Research Failed: {e}")


class USGSRiverGauge:
    def __init__(
        self, site_codes: List[str] = ["01548500", "01531500", "01531000", "01518700"]
    ):
        # Defaults: Pine Creek (Cedar Run), Susquehanna (Towanda), Susquehanna (Waverly NY), Tioga River (Tioga Jct)
        self.site_codes = site_codes

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_levels(self) -> str:
        import httpx

        sites_str = ",".join(self.site_codes)
        # 00060 is Discharge (cfs), 00065 is Gage height (ft)
        url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={sites_str}&parameterCd=00060,00065&siteStatus=all"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                if response.status_code != 200:
                    return f"USGS API Error: {response.status_code}"

                data = response.json()
                results = []

                # Parse USGS JSON structure
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

                # Deduplicate slightly as USGS sends separate timeSeries for cfs and ft
                return "\n".join(sorted(list(set(results))))
        except Exception as e:
            return f"USGS Error: {str(e)}"


class NWSAlerts:
    def __init__(self, lat: float = 41.8, lon: float = -77.1):
        self.lat = lat
        self.lon = lon
        self.headers = {"User-Agent": "RadarTacticalHUD/1.0 (forsythc@ucr.edu)"}

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def get_alerts(self) -> str:
        import httpx

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
        import httpx

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

                # Get the 3 most recently added to the catalog
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
        """Fetch and synthesize headlines from technical news wires."""
        import httpx
        from bs4 import BeautifulSoup

        all_headlines = []
        async with httpx.AsyncClient() as client:
            for url in self.feeds:
                try:
                    response = await client.get(url, timeout=10.0)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, "xml")
                        items = soup.find_all("item")[:3]  # Top 3 from each
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
        """Fetch full content of latest news items and ingest them."""
        import httpx
        from bs4 import BeautifulSoup

        results = []
        async with httpx.AsyncClient() as client:
            for url in self.wire.feeds:
                try:
                    response = await client.get(url, timeout=10.0)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, "xml")
                        items = soup.find_all("item")[
                            :2
                        ]  # Just 2 most recent to avoid bloat
                        for item in items:
                            if item.title and item.link:
                                title = item.title.text.strip()
                                link = item.link.text.strip()
                            else:
                                continue

                            # Fetch full content
                            article_resp = await client.get(link, timeout=15.0)
                            if article_resp.status_code == 200:
                                art_soup = BeautifulSoup(
                                    article_resp.text, "html.parser"
                                )
                                # Simple body extraction
                                text = art_soup.get_text(separator=" ", strip=True)

                                # Process as signal
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
        """Fetch real-time METAR data (legacy station-based)."""
        import httpx

        # Keeping KELM as a secondary source
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
        """Fetch professional weather data using the 'atmos' tool."""
        import subprocess

        try:
            result = subprocess.run(
                ["/home/chuck/.local/bin/atmos", self.location],
                capture_output=True,
                text=True,
                timeout=15.0,
            )
            # Remove ANSI color codes for cleaner ingestion
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

        # Grid Status
        grid = GridScanner()
        grid_status = await grid.get_summary()

        return f"### SECTOR OPS SITREP\n\n#### Professional Weather (Atmos)\n{atmos}\n\n#### Aviation Weather (KELM)\n{metar}\n\n#### Grid Stability (Utility Monitoring)\n{grid_status}\n\n#### Orbital Intelligence (Upcoming Passes)\n{sat_passes}"


class GridScanner:
    def __init__(self):
        self.browser = BrowserIngestAgent()
        self.penelec_url = "https://outages.firstenergycorp.com/pa.html"
        self.tri_county_url = "https://outagemap.tri-countyrec.com/"
        self.nyseg_url = "https://outagemap.nyseg.com/"
        self.ppl_url = "https://omap.pplweb.com/OMAP"

    async def get_summary(self) -> str:
        """Fetch real-time grid status from utility maps across the PA/NY sector."""
        import asyncio

        # Use simple browser extracts for speed
        penelec_task = self.browser.extract(
            self.penelec_url,
            "Extract the total number of customers without power in Tioga and Bradford counties from the Penelec outage summary.",
        )

        tri_county_task = self.browser.extract(
            self.tri_county_url,
            "Identify if there are any active power outages on the map for the Tioga County region. Return a simple status summary.",
        )

        nyseg_task = self.browser.extract(
            self.nyseg_url,
            "Identify the number of power outages or customers affected in Broome and Chemung counties (New York). Return a simple summary.",
        )

        ppl_task = self.browser.extract(
            self.ppl_url,
            "Identify the number of power outages or customers affected in Lackawanna, Dauphin, and Monroe counties (Pennsylvania). Return a simple summary.",
        )

        # Run them concurrently to keep sync times low
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
        # High-Priority SIGINT/Weather/Imaging Targets
        self.targets = {
            "ISS": "25544",
            "NOAA 15": "25338",
            "NOAA 18": "28654",
            "NOAA 19": "33591",
            "METEOR M2-3": "57166",
        }

    async def get_next_passes(self) -> str:
        """Calculate next passes for targeted satellites over the sector."""
        from skyfield.api import Topos, load
        from datetime import datetime, timedelta

        try:
            # Load TLE data
            stations_url = "https://celestrak.org/NORAD/elements/weather.txt"
            stations = load.tle_file(stations_url)
            # Add ISS manually from its own feed
            iss_url = "https://celestrak.org/NORAD/elements/stations.txt"
            stations_iss = load.tle_file(iss_url)

            ts = load.timescale()
            t0 = ts.now()
            t1 = ts.from_datetime(datetime.now() + timedelta(hours=12))

            location = Topos(latitude_degrees=self.lat, longitude_degrees=self.lon)
            results = []

            for name, norad_id in self.targets.items():
                # Find by NORAD ID or close name match
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
                        # Find the next peak (event type 1)
                        for ti, event in zip(t, events):
                            if event == 1:  # Peak of pass
                                # Convert to local time
                                local_time = ti.utc_datetime() - timedelta(
                                    hours=4
                                )  # Simple EDT offset
                                results.append(
                                    f"- **{name}:** {local_time.strftime('%H:%M')} (Alt: {sat.at(ti).subpoint().elevation.km:.0f}km)"
                                )
                                break

            if not results:
                return (
                    "No high-priority satellite passes detected in the next 12 hours."
                )

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
        """Fetch data from all sensors and synthesize a master SITREP with optional delta analysis."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Gather all intel concurrently for speed
        import asyncio

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

        delta_analysis = ""
        if previous_sitrep:
            prompt = f"""
Analyze the following two SITREPs and identify the specific DELTAS (changes). 
Focus on:
1. Shifting river levels or flow rates.
2. Changes in aircraft density or noteworthy new callsigns.
3. New severe weather alerts or cyber vulnerabilities.

PREVIOUS SITREP:
{previous_sitrep[:2000]}

CURRENT DATA:
{adsb_data}
{usgs_data}
{nws_data}

OUTPUT:
Provide a concise, bulleted list of the most significant changes.
"""
            from google import genai

            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            delta_resp = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
            )
            delta_analysis = (
                f"\n## TACTICAL DELTAS (Since Last Update)\n{delta_resp.text}\n"
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

## STRATEGIC CONTEXT
This report represents a comprehensive snapshot of the local tactical environment, including aircraft telemetry, radio frequency activity, hydrological levels for sustainment operations, and critical global cyber threats.
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
            # Run the command asynchronously
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

        # If we have packets from a live session, return them
        if self.packets:
            lines = ["### APRS SITREP - Local Radio Traffic"]
            for p in self.packets[-15:]:
                lines.append(f"- {p}")
            return "\n".join(lines)

        # If running as a standalone snapshot, we need to connect briefly
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), timeout=5.0
            )
            login_str = f"user {self.callsign} pass -1 vers RadarIntelligence 0.1 filter {self.filter_str}\n"
            writer.write(login_str.encode())
            await writer.drain()

            lines = ["### APRS SITREP - Local Radio Traffic"]
            packets_found = False

            # Listen for 3 seconds to catch active traffic
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
            return "No local APRS traffic detected during the 3-second capture window."

        except Exception as e:
            return f"APRS Sensor Error: {e}"

    async def start_stream(self):
        import asyncio

        reader, writer = await asyncio.open_connection(self.host, self.port)

        # Login
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
                    continue  # Skip comments

                self.packets.append(packet_text)
                if len(self.packets) > self.max_packets:
                    self.packets.pop(0)

        finally:
            writer.close()
            await writer.wait_closed()


class BrowserIngestAgent:
    def __init__(self):
        # We assume browser-use is installed globally or in the active miniconda env
        self.cmd_base = [
            "/home/chuck/miniconda3/bin/browser-use",
            "--headless",
            "--model",
            "gemini-3-flash-preview",
        ]

    async def extract(self, url: str, instructions: str) -> str:
        """Use browser-use to dynamically interact with and extract data from a URL."""
        import subprocess
        import asyncio

        prompt = f"Go to {url} and {instructions}. Output only the final extracted data, nothing else."
        cmd = self.cmd_base + ["-p", prompt]

        try:
            # We use asyncio.to_thread because subprocess.run is blocking,
            # and browser-use can take a while to navigate and extract.
            def run_browser():
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    return f"Browser Agent Error: {result.stderr}"

                # Try to find the "Result:" block in the stdout to clean it up
                output = result.stdout
                if "📄 Result:" in output:
                    return output.split("📄 Result:")[1].strip()
                return output.strip()

            return await asyncio.to_thread(run_browser)

        except Exception as e:
            return f"Browser Agent Exception: {e}"


class WebIngestAgent:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        self.intel = IntelligenceAgent()

    async def fetch(self, url: str) -> str:
        """Fetch HTML content from a URL."""
        async with httpx.AsyncClient(
            headers=self.headers, follow_redirects=True
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.text

    async def ingest(self, url: str) -> Tuple[Signal, KnowledgeGraphExtraction]:
        """Ingest a market signal from a URL."""
        html = await self.fetch(url)
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text(separator=" ", strip=True)
        # Prepend metadata for the LLM
        full_text = f"Source URL: {url}\nDate: {datetime.now()}\n\n{text}"

        signal, kg = await self.intel.parse(full_text)
        signal.url = url
        return signal, kg


class TextIngestAgent:
    def __init__(self):
        self.intel = IntelligenceAgent()

    async def ingest(self, text: str) -> Tuple[Signal, KnowledgeGraphExtraction]:
        """Ingest raw text from any source."""
        logger.info("Ingesting raw text from stdin")
        signal, kg = await self.intel.parse(text)
        logger.info(f"Parsed signal: {signal.title}")
        return signal, kg
