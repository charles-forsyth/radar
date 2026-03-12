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
from radar.db.engine import async_session

logger = logging.getLogger(__name__)


class IntelligenceAgent:
    def __init__(self):
        self.embed_bin = "src/radar/tools/radar_embed"
        self.extract_bin = "src/radar/tools/radar_extract"
        self.summarize_bin = "src/radar/tools/radar_summarize"
        self.fetch_bin = "src/radar/tools/radar_fetch"

    async def get_embedding(self, text: str) -> List[float]:
        output = self._run_tool(self.embed_bin, text)
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return [0.0] * 3072

    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [await self.get_embedding(t) for t in texts]

    async def search_signals(self, query: str, limit: int = 5) -> List[Signal]:
        """Hybrid Search using local BM25s, prioritized by date with deduplication."""
        from sqlalchemy import select, desc
        import bm25s
        import logging
        logging.getLogger("bm25s").setLevel(logging.WARNING)

        async with async_session() as session:
            stmt = select(Signal).order_by(desc(Signal.date)).limit(500)
            results = await session.execute(stmt)
            signals = results.scalars().all()
            
            if not signals:
                return []
                
            # Deduplicate by title before indexing to prevent spamming the results
            seen_titles = set()
            unique_signals = []
            for s in signals:
                if s.title not in seen_titles:
                    seen_titles.add(s.title)
                    unique_signals.append(s)
            
            corpus = [f"{s.title}\n{s.content}" for s in unique_signals]
            retriever = bm25s.BM25()
            corpus_tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)
            retriever.index(corpus_tokens, show_progress=False)
            # Query
            query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
            doc_indices, scores = retriever.retrieve(query_tokens, k=min(limit, len(unique_signals)), show_progress=False)

            # Apply time decay penalty to prioritize fresh data
            import numpy as np
            from datetime import timezone

            now = datetime.now(timezone.utc)
            adjusted_scores = []

            for i, idx in enumerate(doc_indices[0]):
                base_score = scores[0][i]
                
                # Filter out completely irrelevant documents or very weak matches
                # In BM25, scores < 1.0 usually mean a very weak single-word hit on a common stop-ish word
                if base_score < 1.0:
                    continue
                    
                sig = unique_signals[int(idx)]
                # Ensure timezone awareness for subtraction
                sig_date = sig.date.replace(tzinfo=timezone.utc) if sig.date.tzinfo is None else sig.date
                age_hours = (now - sig_date).total_seconds() / 3600.0

                # Decay factor: halve the score every 24 hours
                decay = np.exp(-np.log(2) * age_hours / 24.0)

                # Apply a massive boost if it's a TACTICAL SITREP, since that is usually what users want when asking for "situation"
                # And check if the query specifically asks for 'tactical'
                is_tactical = 10.0 if ("TACTICAL SITREP" in sig.title.upper() and "tactical" in query.lower()) else 1.0

                final_score = base_score * decay * is_tactical
                adjusted_scores.append((final_score, sig))

            # Sort by new adjusted score
            adjusted_scores.sort(key=lambda x: x[0], reverse=True)

            return [s for score, s in adjusted_scores]

    def _run_tool(self, tool: str, text: str) -> str:
        try:
            result = subprocess.run(
                [tool], input=text.encode('utf-8', errors='ignore'), capture_output=True, check=True
            )
            return result.stdout.decode('utf-8', errors='ignore').strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running {tool}: {e.stderr.decode('utf-8', errors='ignore')}")
            return ""

    def _fetch_url(self, url: str) -> str:
        try:
            is_pdf = url.lower().endswith(".pdf")
            result = subprocess.run([self.fetch_bin, url], capture_output=True, check=True)
            if is_pdf:
                import os, tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(result.stdout)
                    tmp_path = tmp.name
                try:
                    pdf_result = subprocess.run(["pdftotext", "-layout", tmp_path, "-"], capture_output=True, text=True, check=True)
                    return pdf_result.stdout.strip()
                except Exception as pdf_e:
                    logger.error(f"PDF extraction failed for {url}: {pdf_e}")
                    return ""
                finally:
                    if os.path.exists(tmp_path): os.remove(tmp_path)
            return result.stdout.decode("utf-8", errors="ignore").strip()
        except subprocess.CalledProcessError as e:
            logger.error(f"Error fetching {url}: {e.stderr.decode('utf-8', errors='ignore')}")
            return ""

    async def parse(self, text: str) -> Tuple[Signal, KnowledgeGraphExtraction]:
        output = self._run_tool(self.extract_bin, text)
        try:
            kg_data = json.loads(output)
            kg = KnowledgeGraphExtraction(**kg_data)
        except Exception as e:
            logger.error(f"Parse error: {e}")
            kg = KnowledgeGraphExtraction(entities=[], connections=[], trends=[])
        signal = Signal(title=text.split("\n")[0][:255], content=text, source="local_nlp", date=datetime.now())
        return signal, kg

    async def answer_question(self, question: str, context_signals: List[Signal]) -> str:
        context_text = "\n\n".join([f"--- Signal: {s.title} ---\n{s.content}" for s in context_signals])
        return self._run_tool(self.summarize_bin, f"Question: {question}\nContext: {context_text}")

    async def optimize_knowledge(self, items: List[dict]) -> dict:
        return {"unified_name": "Local Merged Entity", "unified_description": "Merged by local tool", "merged_ids": []}

    async def generate_report(self, topic: str, context: dict) -> str:
        summary = self._run_tool(self.summarize_bin, str(context))
        return f"# Report on {topic}\n\n{summary}"

    async def generate_briefing(self, context: dict) -> str:
        return self._run_tool(self.summarize_bin, str(context))

    async def check_watchpoints(self, signal_content: str, watchpoints: List[dict]) -> List[dict]:
        matches = []
        content_lower = signal_content.lower()
        for wp in watchpoints:
            if str(wp.get("id", "")).lower() in content_lower:
                matches.append({"watchpoint_id": wp.get("id"), "reason": "Local keyword match"})
        return matches

    async def detect_anomalies(self, current_sitrep: str, baseline_context: str) -> List[dict]:
        import numpy as np
        from sklearn.ensemble import IsolationForest
        anomalies = []
        def extract_features(text: str) -> np.ndarray:
            t = text.lower()
            return np.array([len(text), t.count("alert"), t.count("fire"), t.count("police"), t.count("emergency"), t.count("warning"), t.count("critical")])
        historical_texts = [s for s in baseline_context.split("--- SITREP") if len(s.strip()) > 50]
        if len(historical_texts) < 5:
            if "critical" in current_sitrep.lower():
                anomalies.append({"domain": "GENERAL", "severity": "CRITICAL", "message": "Manual keyword match (insufficient baseline)."})
            return anomalies
        X_train = np.array([extract_features(t) for t in historical_texts])
        X_current = extract_features(current_sitrep).reshape(1, -1)
        clf = IsolationForest(random_state=42, contamination=0.1)
        clf.fit(X_train)
        if clf.predict(X_current)[0] == -1:
            score = clf.score_samples(X_current)[0]
            anomalies.append({"domain": "TACTICAL_ML", "severity": "CRITICAL" if score < -0.6 else "WARNING", "message": f"Statistical anomaly detected. Score: {score:.3f}"})
        return anomalies

    async def chat(self, question: str, context_signals: List[Signal], history: List[dict] = []) -> str:
        return await self.answer_question(question, context_signals)


class DeepResearchAgent:
    def __init__(self):
        self.intel = IntelligenceAgent()
        self.sources = ["https://www.hpcwire.com/", "https://www.rtl-sdr.com/", "https://www.hackaday.com/", "https://thehackernews.com/", "https://www.phoronix.com/", "https://www.radioreference.com/"]

    async def research(self, topic: str) -> str:
        from ddgs import DDGS
        from playwright.async_api import async_playwright
        import asyncio
        logger.info(f"Researching: {topic}")
        urls = []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(topic, max_results=3))
                urls = [r["href"] for r in results if "href" in r]
        except Exception: pass
        if not urls: urls = self.sources
        
        combined_text = ""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            for url in urls:
                try:
                    if url.lower().endswith(".pdf"):
                        content = self.intel._fetch_url(url)
                    else:
                        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                        await asyncio.sleep(3)
                        content = await page.evaluate("() => document.body.innerText")
                    if content: combined_text += f"\n--- Source: {url} ---\n{content[:5000]}"
                except Exception: continue
            await browser.close()
        if not combined_text: return f"No hits on {topic}"
        summary = self.intel._run_tool(self.intel.summarize_bin, f"Question: {topic}\n{combined_text}")
        return f"Autonomous Research Report for: {topic}\n\n{summary}"


class BrowserIngestAgent:
    def __init__(self):
        self.intel = IntelligenceAgent()

    async def extract(self, url: str, instructions: str) -> str:
        from playwright.async_api import async_playwright
        import asyncio, os, tempfile
        content = ""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
                await page.goto(url, wait_until="networkidle", timeout=40000)
                await asyncio.sleep(2)
                try:
                    # Dismiss modals
                    await page.get_by_role("button", name="Got it").click(timeout=2000)
                    await asyncio.sleep(1)
                except Exception: pass
                
                # Precise table parsing for Broadcastify
                if "broadcastify.com" in url:
                    try:
                        await page.wait_for_selector(".btable", timeout=15000)
                        extracted_feeds = await page.evaluate("""() => {
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
                    except Exception as bcf_err:
                        logger.error(f"Broadcastify table extraction failed: {bcf_err}")
                        extracted_feeds = []
                        
                    if extracted_feeds:
                        content = "BROADCASTIFY LIVE FEED DATA:\n" + "\n".join(extracted_feeds)
                    else:
                        content = await page.evaluate("() => document.body.innerText")
                else:
                    content = await page.evaluate("() => document.body.innerText")
                
                if (len(content) < 500 or "map" in url.lower()) and "ocr" in instructions.lower():
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        spath = tmp.name
                    await page.screenshot(path=spath, full_page=True)
                    try:
                        import pytesseract
                        from PIL import Image
                        content += f"\n--- OCR EXTRACTION ---\n{pytesseract.image_to_string(Image.open(spath))}"
                    except Exception: pass
                    finally:
                        if os.path.exists(spath): os.remove(spath)
                await browser.close()
        except Exception as e:
            logger.error(f"Playwright dynamic extraction failed for {url}: {e}")
            content = self.intel._fetch_url(url)
        return self.intel._run_tool(self.intel.summarize_bin, f"Question: {instructions}\nTarget: {url}\n\n{content[:15000]}")


class WebIngestAgent:
    def __init__(self):
        self.intel = IntelligenceAgent()
    async def fetch(self, url: str) -> str:
        return self.intel._fetch_url(url)
    async def ingest(self, url: str) -> Tuple[Signal, KnowledgeGraphExtraction]:
        text = await self.fetch(url)
        full_text = f"Source URL: {url}\nDate: {datetime.now()}\n\n{text}"
        signal, kg = await self.intel.parse(full_text)
        signal.url = url
        return signal, kg


class TextIngestAgent:
    def __init__(self):
        self.intel = IntelligenceAgent()
    async def ingest(self, text: str) -> Tuple[Signal, KnowledgeGraphExtraction]:
        signal, kg = await self.intel.parse(text)
        return signal, kg


class AudioIngestAgent:
    def __init__(self):
        self.intel = IntelligenceAgent()
    async def ingest_stream(self, stream_url: str, duration_sec: int = 15) -> str:
        import os, tempfile, asyncio
        from faster_whisper import WhisperModel
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            apath = tmp.name
        try:
            p = await asyncio.create_subprocess_exec("ffmpeg", "-y", "-i", stream_url, "-t", str(duration_sec), "-f", "mp3", apath, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await p.communicate()
            model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(apath, beam_size=1)
            trans = " ".join([s.text for s in segments]).strip()
            return trans if trans else "No voice detected."
        except Exception as e: return f"Error: {e}"
        finally:
            if os.path.exists(apath): os.remove(apath)


class USGSRiverGauge:
    def __init__(self, site_codes: List[str] = ["01548500", "01531500", "01531000", "01518700"]):
        self.site_codes = site_codes
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_levels(self) -> str:
        url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={','.join(self.site_codes)}&parameterCd=00060,00065&siteStatus=all"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url)
            if resp.status_code != 200: return f"Error: {resp.status_code}"
            data = resp.json()
            res = []
            for ts in data.get("value", {}).get("timeSeries", []):
                name = ts["sourceInfo"]["siteName"]
                val = ts["values"][0]["value"][0]["value"]
                unit = "ft" if "height" in ts["variable"]["variableName"].lower() else "cfs"
                res.append(f"- {name}: {val} {unit}")
            return "\n".join(sorted(list(set(res)))) if res else "No data."


class NWSAlerts:
    def __init__(self, lat: float = 41.8, lon: float = -77.1):
        self.lat, self.lon = lat, lon
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_alerts(self) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"https://api.weather.gov/alerts/active?point={self.lat},{self.lon}", headers={"User-Agent": "Radar/1.0"})
            if resp.status_code != 200: return f"Error: {resp.status_code}"
            feats = resp.json().get("features", [])
            return "\n".join([f"⚠️ {f['properties']['headline']}" for f in feats]) if feats else "No alerts."


class CISAFeed:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_latest_vulns(self) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.get("https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json")
            if resp.status_code != 200: return f"Error: {resp.status_code}"
            v = sorted(resp.json().get("vulnerabilities", []), key=lambda x: x.get("dateAdded", ""), reverse=True)[:3]
            return "\n".join([f"- {i['cveID']}: {i['vulnerabilityName']}" for i in v]) if v else "No data."


class NewsWire:
    def __init__(self):
        self.feeds = ["https://www.hackaday.com/blog/feed/", "https://www.rtl-sdr.com/feed/", "https://www.hpcwire.com/feed/", "https://feeds.feedburner.com/TheHackersNews"]
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_headlines(self) -> str:
        h = []
        async with httpx.AsyncClient() as client:
            for url in self.feeds:
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        soup = BeautifulSoup(r.text, "xml")
                        h.extend([f"- {i.title.text.strip()}" for i in soup.find_all("item")[:3] if i.title])
                except Exception: continue
        return "\n".join(h[:10]) if h else "No news."


class RSSIngestAgent:
    def __init__(self):
        self.wire = NewsWire()
        self.intel = IntelligenceAgent()
    async def sync_news(self) -> List[Tuple[Signal, KnowledgeGraphExtraction]]:
        res = []
        async with httpx.AsyncClient() as client:
            for url in self.wire.feeds:
                try:
                    r = await client.get(url)
                    if r.status_code == 200:
                        soup = BeautifulSoup(r.text, "xml")
                        for i in soup.find_all("item")[:2]:
                            if i.title and i.link:
                                ar = await client.get(i.link.text.strip())
                                if ar.status_code == 200:
                                    txt = BeautifulSoup(ar.text, "html.parser").get_text(separator=" ", strip=True)
                                    s, kg = await self.intel.parse(f"Title: {i.title.text.strip()}\n\n{txt[:10000]}")
                                    res.append((s, kg))
                except Exception: continue
        return res


class SectorScanner:
    def __init__(self, loc: str = "Tioga County, PA"):
        self.loc = loc
    async def get_metar(self) -> str:
        async with httpx.AsyncClient() as client:
            r = await client.get("https://aviationweather.gov/api/data/metar?ids=KELM")
            return r.text.strip() if r.status_code == 200 else "Error."
    async def get_atmos_weather(self) -> str:
        try:
            r = subprocess.run(["/home/chuck/.local/bin/atmos", self.loc], capture_output=True, text=True)
            import re
            return re.sub(r"\x1b\[[0-9;]*m", "", r.stdout).strip()
        except Exception: return "Error."


class GridScanner:
    def __init__(self):
        self.browser = BrowserIngestAgent()
    async def get_summary(self) -> str:
        import asyncio
        urls = ["https://outages.firstenergycorp.com/pa.html", "https://outagemap.tri-countyrec.com/", "https://outagemap.nyseg.com/", "https://omap.pplweb.com/OMAP"]
        tasks = [self.browser.extract(u, "Extract outages") for u in urls]
        res = await asyncio.gather(*tasks)
        return "\n".join([f"- {u}: {r[:100]}..." for u, r in zip(urls, res)])


class SatelliteScanner:
    def __init__(self, lat: float = 41.8, lon: float = -77.1):
        self.lat, self.lon = lat, lon
        self.targets = {"ISS": "25544", "NOAA 15": "25338", "NOAA 18": "28654", "NOAA 19": "33591", "METEOR M2-3": "57166"}
    async def get_next_passes(self) -> str:
        from skyfield.api import Topos, load
        from datetime import datetime, timedelta
        try:
            sats = load.tle_file("https://celestrak.org/NORAD/elements/weather.txt") + load.tle_file("https://celestrak.org/NORAD/elements/stations.txt")
            ts, t0 = load.timescale(), load.timescale().now()
            t1 = ts.from_datetime(datetime.now() + timedelta(hours=12))
            loc = Topos(latitude_degrees=self.lat, longitude_degrees=self.lon)
            res = []
            for name, nid in self.targets.items():
                s = next((x for x in sats if nid in x.name or name in x.name), None)
                if s:
                    t, events = s.find_events(loc, t0, t1, altitude_degrees=10.0)
                    for ti, event in zip(t, events):
                        if event == 1:
                            lt = ti.utc_datetime() - timedelta(hours=4)
                            res.append(f"- **{name}:** {lt.strftime('%H:%M')} (Alt: {s.at(ti).subpoint().elevation.km:.0f}km)")
                            break
            return "\n".join(res) if res else "None."
        except Exception as e: return f"Error: {e}"


class LocalSoftwareScanner:
    async def get_summary(self) -> str:
        import subprocess
        import asyncio

        async def run_count(cmd: str, adjust: int = 0) -> str:
            try:
                proc = await asyncio.create_subprocess_shell(
                    cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await proc.communicate()
                if proc.returncode == 0:
                    val = stdout.decode().strip()
                    if not val:
                        return "0"
                    count = int(val) - adjust
                    return str(count) if count >= 0 else "0"
                return "Error"
            except Exception:
                return "Not Found"

        # Gather software and container counts concurrently
        (
            dpkg_count, 
            pip_count, 
            uv_count, 
            mm_count,
            docker_ps,
            docker_img,
            singularity_count,
            gh_repos
        ) = await asyncio.gather(
            run_count("dpkg -l | wc -l", adjust=5),
            run_count("/home/chuck/bin/pip list | wc -l", adjust=2),
            run_count("/home/chuck/.local/bin/uv pip list | wc -l"),
            run_count("/home/chuck/bin/micromamba list | wc -l", adjust=4),
            run_count("docker ps -q | wc -l"),
            run_count("docker images -q | wc -l"),
            run_count("find ~ -maxdepth 3 -type f -name '*.sif' -o -name '*.simg' 2>/dev/null | wc -l"),
            run_count("gh repo list --limit 1000 | wc -l")
        )

        return (
            f"### LOCAL SYSTEM SOFTWARE INVENTORY\n"
            f"- **APT (dpkg):** {dpkg_count} installed packages\n"
            f"- **pip (Global):** {pip_count} installed packages\n"
            f"- **uv (Local Env):** {uv_count} installed packages\n"
            f"- **micromamba (Base):** {mm_count} installed packages\n"
            f"### CONTAINERS & REPOSITORIES\n"
            f"- **Docker:** {docker_ps} running containers / {docker_img} total images\n"
            f"- **Singularity/Apptainer:** {singularity_count} local .sif/.simg images\n"
            f"- **GitHub Repositories:** {gh_repos} tracked remote repos"
        )


class WidebandSDRScanner:
    def __init__(self, host: str = "192.168.1.246", user: str = "pi"):
        self.host, self.user = host, user

    async def get_snapshot_text(self) -> str:
        import asyncio, json
        # Execute the juggler script on CORE via SSH
        cmd = ["ssh", f"{self.user}@{self.host}", "python3 /home/pi/bin/radar_sdr_juggler.py"]
        try:
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                return f"Wideband SDR Error: {stderr.decode()}"
            
            data = json.loads(stdout.decode())
            if "error" in data:
                return f"Wideband SDR Error: {data['error']}"
            
            signals = data.get("top_signals", [])
            if not signals:
                return "No strong RF signals detected in wideband sweep."
                
            lines = ["### FULL SPECTRUM RF SWEEP (1MHz - 1700MHz)"]
            for s in signals:
                lines.append(f"- Frequency: {s['freq']:.2f} MHz | Power: {s['db']:.2f} dB")
            return "\n".join(lines)
        except Exception as e:
            return f"Wideband SDR Exception: {str(e)}"

class NetworkAndSecurityScanner:
    async def get_summary(self) -> str:
        import subprocess
        import asyncio
        import re

        async def run_cmd(cmd: str) -> str:
            try:
                proc = await asyncio.create_subprocess_shell(
                    cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, _ = await proc.communicate()
                return stdout.decode().strip() if proc.returncode == 0 else ""
            except Exception:
                return ""

        # 1. ARP Scan (Local LAN devices)
        arp_output = await run_cmd("sudo -n arp-scan -l")
        device_count = 0
        vendors = set()
        if arp_output:
            for line in arp_output.split("\n"):
                if re.match(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+", line):
                    device_count += 1
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        vendor = parts[2].strip()
                        if not vendor.startswith("(Unknown"):
                            vendors.add(vendor.split(" ")[0].replace(",", ""))

        # 2. Ping Latency (Internet Health)
        ping_output = await run_cmd("ping -c 3 1.1.1.1")
        latency = "Offline"
        if ping_output:
            match = re.search(r"min/avg/max/mdev = [\d\.]+/([\d\.]+)/", ping_output)
            if match:
                latency = f"{match.group(1)} ms"

        # 3. Failed SSH/Intrusions
        auth_output = await run_cmd("sudo -n grep 'Failed password' /var/log/auth.log | wc -l")
        failed_logins = auth_output if auth_output else "0"

        vendors_str = ", ".join(list(vendors)[:5]) if vendors else "None identified"

        return (
            f"### NETWORK & SECURITY INTEGRITY\n"
            f"- **Internet Latency (1.1.1.1):** {latency}\n"
            f"- **Local LAN Devices (ARP):** {device_count} active devices\n"
            f"- **Identified Hardware:** {vendors_str}\n"
            f"- **Failed SSH Logins (Auth):** {failed_logins} recent attempts"
        )


class TacticalAgent:
    def __init__(self):
        self.adsb, self.aprs, self.sector, self.usgs, self.nws, self.cisa, self.software, self.rf_sweep, self.netsec = (
            ADSBScanner(),
            APRSStreamer(),
            SectorScanner(),
            USGSRiverGauge(),
            NWSAlerts(),
            CISAFeed(),
            LocalSoftwareScanner(),
            WidebandSDRScanner(),
            NetworkAndSecurityScanner(),
        )

    async def generate_full_sitrep(self, prev: Optional[str] = None) -> str:
        import asyncio

        data = await asyncio.gather(
            self.adsb.get_snapshot_text(),
            self.sector.get_atmos_weather(),
            self.aprs.get_snapshot_text(),
            self.usgs.get_levels(),
            self.nws.get_alerts(),
            self.cisa.get_latest_vulns(),
            self.software.get_summary(),
            self.rf_sweep.get_snapshot_text(),
            self.netsec.get_summary(),
        )
        return f"Title: Master Tactical SITREP - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{data[1]}\n{data[4]}\n{data[8]}\n{data[3]}\n{data[6]}\n{data[0]}\n{data[7]}\n{data[2]}\n{data[5]}"


class ADSBScanner:
    def __init__(self, host: str = "192.168.1.246", user: str = "pi"):
        self.host, self.user = host, user
    async def get_live_data(self) -> dict:
        import asyncio, json
        p = await asyncio.create_subprocess_exec("ssh", f"{self.user}@{self.host}", "cat /home/pi/adsb_data/aircraft.json", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await p.communicate()
        return json.loads(out.decode()) if p.returncode == 0 else {"error": err.decode()}
    async def get_snapshot_text(self) -> str:
        d = await self.get_live_data()
        if "error" in d: return f"Error: {d['error']}"
        ac = d.get("aircraft", [])
        return "\n".join([f"- Flight {x.get('flight','Unk').strip()} at {x.get('alt_baro',0)}ft" for x in ac]) if ac else "No aircraft."


class APRSStreamer:
    def __init__(self, host: str = "noam.aprs2.net", port: int = 14580, callsign: str = "NOCALL", filter: str = "r/41.8/-77.1/100"):
        self.host, self.port, self.callsign, self.filter = host, port, callsign, filter
        self.packets = []
    async def get_snapshot_text(self) -> str:
        return "\n".join([f"- {p}" for p in self.packets[-15:]]) if self.packets else "No traffic."
    async def start_stream(self):
        import asyncio
        r, w = await asyncio.open_connection(self.host, self.port)
        w.write(f"user {self.callsign} pass -1 vers Radar 0.1 filter {self.filter}\n".encode())
        await w.drain()
        try:
            while True:
                line = await r.readline()
                if not line: break
                p = line.decode().strip()
                if not p.startswith("#"):
                    self.packets.append(p)
                    if len(self.packets) > 50: self.packets.pop(0)
        finally: w.close(); await w.wait_closed()
