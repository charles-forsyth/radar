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

logger = logging.getLogger(__name__)


class IntelligenceAgent:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def get_embedding(self, text: str) -> List[float]:
        """Generate a vector embedding for the given text."""
        result = self.client.models.embed_content(
            model=settings.EMBEDDING_MODEL,
            contents=text,
        )
        return result.embeddings[0].values

    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a batch of texts."""
        if not texts:
            return []

        # We process embeddings sequentially or concurrently here
        # Optimizing with `batch_embed_contents` is better but we use a loop for type safety for now

        embeddings = []
        for t in texts:
            result = self.client.models.embed_content(
                model=settings.EMBEDDING_MODEL,
                contents=t,
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    async def extract_knowledge(self, text: str) -> KnowledgeGraphExtraction:
        """Extract entities, relationships, and trends from text using structured generation."""
        prompt = """
        Analyze the following text and extract a Knowledge Graph.
        
        1. **Entities:** Identify key players (Companies, People) and Technologies.
        2. **Connections:** Map the relationships between them (who is competing with whom, who supports what).
        3. **Trends:** Identify broader market trends or patterns (e.g., "AI Consolidation", "Green Tech Surge"). Estimate their velocity.
        
        Text:
        {text}
        """

        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt.format(text=text),
            config={
                "response_mime_type": "application/json",
                "response_schema": KnowledgeGraphExtraction,
            },
        )

        try:
            # The response.text should be JSON conforming to the schema
            return KnowledgeGraphExtraction.model_validate_json(response.text)
        except Exception as e:
            logger.error(f"Failed to parse knowledge extraction: {e}")
            return KnowledgeGraphExtraction(entities=[], connections=[], trends=[])

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
You are IVXXa, the Captain's second in command. Provide a sophisticated, inclusive, and highly insightful 90-second verbal intelligence briefing.

RECENT ACTIVITY (LAST 24 HOURS):
- Strategic Signals: {context.get("signals", [])}
- Emerging Trends: {context.get("trends", [])}
- Tactical Sensors (Air/RF/Weather): {context.get("tactical", "No tactical data captured.")}
- Global News Wire: {context.get("news", "No news signals.")}

INSTRUCTIONS:
1. Start with "Captain, IVXXa reporting. Here is your synthesized intelligence briefing."
2. DO NOT just list data. Perform "DOT CONNECTING":
   - How do the global tech trends impact our local tactical posture?
   - Are there correlations between the weather/hydrology and our field readiness?
   - How do the latest cyber threats impact our regional infrastructure?
3. Maintain a tone of absolute loyalty, strategic foresight, and operational urgency.
4. Ensure the briefing is inclusive of all data domains: Global Strategy, Local Tactics, and Tech/SDR developments.
5. End with a "Platform Readiness" status.

Total length: 250-300 words. Be sharp.
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

        response = self.client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": List[WatchMatch],
            },
        )
        import json

        return json.loads(response.text)

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


class USGSRiverGauge:
    def __init__(
        self, site_codes: List[str] = ["01548500", "01531500", "01531000", "01518700"]
    ):
        # Defaults: Pine Creek (Cedar Run), Susquehanna (Towanda), Susquehanna (Waverly NY), Tioga River (Tioga Jct)
        self.site_codes = site_codes

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

    async def get_alerts(self) -> str:
        import httpx

        url = f"https://api.weather.gov/alerts/active?point={self.lat},{self.lon}"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=self.headers)
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
    async def get_latest_vulns(self) -> str:
        import httpx

        url = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
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

    async def get_headlines(self) -> str:
        """Fetch and synthesize headlines from technical news wires."""
        import httpx
        from bs4 import BeautifulSoup

        all_headlines = []
        async with httpx.AsyncClient() as client:
            for url in self.feeds:
                try:
                    response = await client.get(url, timeout=5.0)
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


class SectorScanner:
    def __init__(self, location: str = "Tioga County, PA"):
        self.location = location

    async def get_metar(self) -> str:
        """Fetch real-time METAR data (legacy station-based)."""
        import httpx

        # Keeping KELM as a secondary source
        url = "https://aviationweather.gov/api/data/metar?ids=KELM"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
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
                ["atmos", self.location], capture_output=True, text=True
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
        return f"### SECTOR OPS SITREP\n\n#### Professional Weather (Atmos)\n{atmos}\n\n#### Aviation Weather (KELM)\n{metar}\n\n- **Satellite Status:** Tracking operational (Predictor implementation pending)"


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
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url, headers=self.headers, follow_redirects=True
                )
                response.raise_for_status()
                return response.text
            except httpx.HTTPError as e:
                logger.error(f"HTTP Error fetching {url}: {e}")
                raise

    async def parse(
        self, html: str, url: str
    ) -> Tuple[Signal, KnowledgeGraphExtraction]:
        soup = BeautifulSoup(html, "lxml")
        title = soup.title.string if soup.title else "No Title"

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator="\n", strip=True)
        content = text[:5000]

        # Generate vector embedding
        vector = await self.intel.get_embedding(content)

        # Extract Knowledge Graph
        kg = await self.intel.extract_knowledge(content)

        signal = Signal(
            title=title,
            url=url,
            content=content,
            raw_text=text,
            date=datetime.now(),
            source="web",
            vector=vector,
        )
        return signal, kg

    async def ingest(self, url: str) -> Tuple[Signal, KnowledgeGraphExtraction]:
        logger.info(f"Ingesting {url}")
        html = await self.fetch(url)
        signal, kg = await self.parse(html, url)
        logger.info(f"Parsed signal: {signal.title}")
        return signal, kg


class TextIngestAgent:
    def __init__(self):
        self.intel = IntelligenceAgent()

    async def parse(
        self, text: str, title: str = "Raw Input"
    ) -> Tuple[Signal, KnowledgeGraphExtraction]:
        # Generate a generic title if not provided or just use the first line
        if title == "Raw Input" and text.strip():
            first_line = text.strip().split("\n")[0][:50]
            if first_line:
                title = first_line

        content = text
        # Generate vector embedding
        vector = await self.intel.get_embedding(content[:5000])

        # Extract Knowledge Graph
        kg = await self.intel.extract_knowledge(content[:5000])

        signal = Signal(
            title=title,
            content=content,
            raw_text=text,
            date=datetime.now(),
            source="stdin",
            vector=vector,
        )
        return signal, kg

    async def ingest(self, text: str) -> Tuple[Signal, KnowledgeGraphExtraction]:
        logger.info("Ingesting raw text from stdin")
        signal, kg = await self.parse(text)
        logger.info(f"Parsed signal: {signal.title}")
        return signal, kg
