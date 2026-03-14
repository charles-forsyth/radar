import json
import logging
import subprocess
import asyncio
import re
from datetime import datetime
from typing import List, Tuple, Optional
import httpx
import trafilatura

from radar.db.models import Signal
from radar.core.models import KnowledgeGraphExtraction, TacticalSnapshot
from radar.db.engine import async_session
from radar.config import settings

logger = logging.getLogger(__name__)


class IntelligenceAgent:
    def __init__(self, intel: Optional["IntelligenceAgent"] = None):
        self.embed_bin = settings.TOOL_EMBED
        self.extract_bin = settings.TOOL_EXTRACT
        self.summarize_bin = settings.TOOL_SUMMARIZE
        self.fetch_bin = settings.TOOL_FETCH
        self.embedding_model = None

    async def get_embedding(self, text: str) -> List[float]:
        """Return dummy vector for legacy schema compatibility."""
        return [0.0] * 384

    def extract_stats(self, text: str) -> List[dict]:
        """High-fidelity tactical OSINT/SIGINT numerical extraction engine with positional context."""
        import re

        stats = []

        def get_subject(match_pos, full_text):
            """Heuristic to find the subject/noun phrase immediately before a specific match position."""
            pre = (
                full_text[max(0, match_pos - 80) : match_pos].replace("\n", " ").strip()
            )
            parts = re.split(r"[,.;:]", pre)
            chunk = parts[-1].strip()
            chunk = re.sub(
                r"^(the|a|an|of|to|for|is|are|was|were|has|been|which|that|this|these)\s+",
                "",
                chunk,
                flags=re.IGNORECASE,
            )
            chunk = re.sub(
                r"\s+(rose|fell|dropped|increased|decreased|at|of|to|is|with|by|around|nearly|about)$",
                "",
                chunk,
                flags=re.IGNORECASE,
            )
            words = chunk.split()
            if len(words) > 5:
                chunk = " ".join(words[-5:])
            return chunk.title() if len(chunk) > 3 else None

        def get_context(match_pos, match_len, full_text, window=80):
            start = max(0, match_pos - window)
            end = min(len(full_text), match_pos + match_len + window)
            return full_text[start:end].replace("\n", " ").strip()

        # 1. SPECIALIZED SIGINT
        for m in re.finditer(r"(-\d+\.?\d*)\s*dBm", text, re.IGNORECASE):
            val = m.group(1)
            subject = get_subject(m.start(), text)
            stats.append(
                {
                    "label": subject if subject else "Noise Floor/RSSI",
                    "value": float(val),
                    "unit": "dBm",
                    "description": get_context(m.start(), len(m.group(0)), text),
                }
            )

        for m in re.finditer(
            r"(\d+\.?\d*)\s*(kHz|MHz|GHz)\s*(?:bandwidth|BW|channel spacing)",
            text,
            re.IGNORECASE,
        ):
            val, unit = m.group(1), m.group(2)
            subject = get_subject(m.start(), text)
            stats.append(
                {
                    "label": subject if subject else "Signal BW",
                    "value": float(val),
                    "unit": unit,
                    "description": get_context(m.start(), len(m.group(0)), text),
                }
            )

        for m in re.finditer(
            r"(\d+)-bit\s*(?:AES|DES|encryption|key)", text, re.IGNORECASE
        ):
            val = m.group(1)
            subject = get_subject(m.start(), text)
            stats.append(
                {
                    "label": subject if subject else "Encryption Depth",
                    "value": float(val),
                    "unit": "bit",
                    "description": get_context(m.start(), len(m.group(0)), text),
                }
            )

        # 2. TACTICAL OSINT
        tactical_patterns = [
            (
                r"(\d+,?\d*)\s*(?:troops|personnel|soldiers|combatants)",
                "Troop Count",
                "Units",
            ),
            (
                r"(\d+,?\d*)\s*(?:casualties|killed|wounded|fatalities)",
                "Casualty Count",
                "Units",
            ),
            (
                r"(\d+,?\d*)\s*(?:drones|UAVs|quadcopters|fixed-wing)",
                "Drone Density",
                "Units",
            ),
            (
                r"(\d+,?\d*)\s*(?:cameras|LPRs|ALPRs|surveillance nodes)",
                "Sensor Density",
                "Units",
            ),
            (
                r"(\d+,?\d*)\s*(?:arrests|detained|apprehensions)",
                "Enforcement Count",
                "Units",
            ),
            (
                r"(\d+\.?\d*)\s*(?:mi|km|miles|kilometers)\s*(?:radius|distance|range)",
                "Tactical Range",
                "Dist",
            ),
        ]
        for pattern, default_label, unit in tactical_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                val = m.group(1)
                subject = get_subject(m.start(), text)
                try:
                    stats.append(
                        {
                            "label": subject if subject else default_label,
                            "value": float(val.replace(",", "")),
                            "unit": unit,
                            "description": get_context(
                                m.start(), len(m.group(0)), text
                            ),
                        }
                    )
                except ValueError:
                    continue

        # 3. FINANCIAL OSINT
        for m in re.finditer(
            r"\$(\d+\.?\d*)\s*([MBK]|Million|Billion)", text, re.IGNORECASE
        ):
            val, scale = m.group(1), m.group(2)
            subject = get_subject(m.start(), text)
            try:
                f_val = float(val)
                s = scale.upper()
                if s.startswith("B") or s == "BILLION":
                    f_val *= 1_000_000_000
                elif s.startswith("M") or s == "MILLION":
                    f_val *= 1_000_000
                elif s.startswith("K"):
                    f_val *= 1_000
                stats.append(
                    {
                        "label": subject if subject else "Strategic Value",
                        "value": f_val,
                        "unit": "USD",
                        "description": get_context(m.start(), len(m.group(0)), text),
                    }
                )
            except ValueError:
                continue

        # 4. Standard Heuristics
        for m in re.finditer(
            r"(?:Gas|Fuel|Gasoline):\s*\$([0-9,]+\.?\d*)", text, re.IGNORECASE
        ):
            val = m.group(1)
            stats.append(
                {
                    "label": "Gas Price",
                    "value": float(val.replace(",", "")),
                    "unit": "USD",
                    "description": get_context(m.start(), len(m.group(0)), text),
                }
            )

        for m in re.finditer(r"\$([0-9,]+\.?\d*)", text):
            val = m.group(1)
            subject = get_subject(m.start(), text)
            try:
                f_val = float(val.replace(",", ""))
                if not any(s["value"] == f_val and s["unit"] == "USD" for s in stats):
                    stats.append(
                        {
                            "label": subject if subject else "Price/Value",
                            "value": f_val,
                            "unit": "USD",
                            "description": get_context(
                                m.start(), len(m.group(0)), text
                            ),
                        }
                    )
            except ValueError:
                continue

        for m in re.finditer(r"(\d+\.?\d*)%", text):
            val = m.group(1)
            subject = get_subject(m.start(), text)
            try:
                stats.append(
                    {
                        "label": subject if subject else "Percentage",
                        "value": float(val),
                        "unit": "%",
                        "description": get_context(m.start(), len(m.group(0)), text),
                    }
                )
            except ValueError:
                continue

        unit_patterns = [
            (r"(\d+,?\d*)\s*(?:acres|acre)", "Land Area", "Acres"),
            (r"(\d+,?\d*)\s*(?:gallons|gal)", "Fuel Volume", "Gallons"),
            (r"(\d+,?\d*)\s*(?:t/s|tokens/sec)", "Performance", "t/s"),
            (r"(\d+,?\d*)\s*(?:beds|bed count)", "Medical Capacity", "Beds"),
        ]
        for pattern, default_label, unit in unit_patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                val = m.group(1)
                subject = get_subject(m.start(), text)
                try:
                    stats.append(
                        {
                            "label": subject if subject else default_label,
                            "value": float(val.replace(",", "")),
                            "unit": unit,
                            "description": get_context(
                                m.start(), len(m.group(0)), text
                            ),
                        }
                    )
                except ValueError:
                    continue

        return stats

    async def search_signals(self, query: str, limit: int = 5) -> List[Signal]:
        """Keyword-based relational search (SQLite friendly)."""
        from sqlalchemy import select, or_

        async with async_session() as session:
            stmt = (
                select(Signal)
                .where(
                    or_(
                        Signal.title.ilike(f"%{query}%"),  # type: ignore
                        Signal.content.ilike(f"%{query}%"),  # type: ignore
                    )
                )
                .order_by(Signal.date.desc())  # type: ignore
                .limit(limit)
            )

            results = await session.execute(stmt)
            return list(results.scalars().all())

    def _clean_html(self, html: str) -> str:
        """Use Trafilatura for high-precision content extraction."""
        try:
            result = trafilatura.extract(
                html, include_comments=False, include_tables=True, no_fallback=False
            )
            return result if result else ""
        except Exception:
            return ""

    async def chat(
        self, question: str, context_signals: List[Signal], history: List[dict] = []
    ) -> str:
        """Lightweight chat passthrough."""
        return await self.answer_question(question, context_signals)

    def _run_tool(self, tool: str, text: str) -> str:
        try:
            result = subprocess.run(
                [tool],
                input=text.encode("utf-8", errors="ignore"),
                capture_output=True,
                check=True,
            )
            return result.stdout.decode("utf-8", errors="ignore").strip()
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Error running {tool}: {e.stderr.decode('utf-8', errors='ignore')}"
            )
            return ""

    def _fetch_url(self, url: str) -> str:
        try:
            is_pdf = url.lower().endswith(".pdf")
            result = subprocess.run(
                [self.fetch_bin, url], capture_output=True, check=True
            )
            if is_pdf:
                import os
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(result.stdout)
                    tmp_path = tmp.name
                try:
                    pdf_result = subprocess.run(
                        ["pdftotext", "-layout", tmp_path, "-"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    return pdf_result.stdout.strip()
                except Exception as pdf_e:
                    logger.error(f"PDF extraction failed for {url}: {pdf_e}")
                    return ""
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            return result.stdout.decode("utf-8", errors="ignore").strip()
        except subprocess.CalledProcessError as e:
            logger.error(
                f"Error fetching {url}: {e.stderr.decode('utf-8', errors='ignore')}"
            )
            return ""

    async def parse(self, text: str) -> Tuple[Signal, KnowledgeGraphExtraction]:
        signal = Signal(
            title=text.split("\n")[0][:255],
            content=text,
            source="local_nlp",
            date=datetime.now(),
        )
        return signal, KnowledgeGraphExtraction(entities=[], connections=[], trends=[])

    async def answer_question(
        self, question: str, context_signals: List[Signal]
    ) -> str:
        context_text = "\n\n".join(
            [f"--- Signal: {s.title} ---\n{s.content}" for s in context_signals]
        )
        return self._run_tool(
            self.summarize_bin, f"Question: {question}\nContext: {context_text}"
        )

    async def generate_briefing(self, context: dict) -> str:
        return self._run_tool(self.summarize_bin, str(context))

    async def detect_anomalies(
        self, current_sitrep: str, baseline_context: str
    ) -> List[dict]:
        anomalies = []
        t = current_sitrep.lower()
        if "critical" in t or "emergency" in t or "fire" in t:
            anomalies.append(
                {
                    "domain": "TACTICAL",
                    "severity": "CRITICAL",
                    "message": "Critical keyword detected in situational report.",
                }
            )
        return anomalies


class DeepResearchAgent:
    def __init__(self, intel: Optional[IntelligenceAgent] = None):
        self.intel = intel or IntelligenceAgent()

    async def research(self, topic: str) -> str:
        from playwright.async_api import async_playwright
        import asyncio

        combined_text = f"🎯 {topic}\n"
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                from ddgs import DDGS

                with DDGS() as ddgs:
                    results = list(ddgs.text(topic, max_results=5))
                    urls = [r["href"] for r in results]

                for url in urls:
                    try:
                        if url.lower().endswith(".pdf"):
                            content = self.intel._fetch_url(url)
                        else:
                            await page.goto(
                                url, wait_until="domcontentloaded", timeout=30000
                            )
                            await asyncio.sleep(3)
                            html = await page.content()
                            content = self.intel._clean_html(html)
                            if not content:
                                content = await page.evaluate(
                                    "() => document.body.innerText"
                                )
                        if content:
                            combined_text += (
                                f"\n--- Source: {url} ---\n{content[:5000]}"
                            )
                    except Exception:
                        continue
            finally:
                await browser.close()
        return combined_text


class BrowserIngestAgent:
    def __init__(self, intel: Optional[IntelligenceAgent] = None):
        self.intel = intel or IntelligenceAgent()

    async def extract(self, url: str, instructions: str) -> str:
        from playwright.async_api import async_playwright
        import asyncio

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page()
                await page.goto(url, wait_until="networkidle", timeout=60000)
                await asyncio.sleep(5)

                content = ""
                if "broadcastify.com" in url:
                    try:
                        await page.wait_for_selector(".btable", timeout=15000)
                        extracted_feeds = await page.evaluate(
                            "() => { "
                            "const results = []; "
                            "const rows = document.querySelectorAll('.btable tr'); "
                            "rows.forEach(row => { "
                            "const cells = row.querySelectorAll('td'); "
                            "if (cells.length > 3) { "
                            "const feedName = cells[1].innerText ? cells[1].innerText.trim().replace(/\\n/g, ' - ') : ''; "
                            "const genre = cells[2].innerText ? cells[2].innerText.trim() : ''; "
                            "const listeners = cells[3].innerText ? cells[3].innerText.trim() : ''; "
                            "if (genre.includes('Public Safety') || parseInt(listeners) >= 0) { "
                            "results.push(`Feed: ${feedName} | Genre: ${genre} | Listeners: ${listeners}`); "
                            "} "
                            "} "
                            "}); "
                            "return results; "
                            "}"
                        )
                        content = "BROADCASTIFY LIVE FEED DATA:\n" + "\n".join(
                            extracted_feeds
                        )
                    except Exception:
                        pass

                if not content:
                    html = await page.content()
                    content = self.intel._clean_html(html)
                    if not content:
                        content = await page.evaluate("() => document.body.innerText")
                return content
            finally:
                await browser.close()


class TextIngestAgent:
    def __init__(self, intel: Optional[IntelligenceAgent] = None):
        self.intel = intel or IntelligenceAgent()

    async def ingest(self, text: str) -> Tuple[Signal, KnowledgeGraphExtraction]:
        return await self.intel.parse(text)


class RSSIngestAgent:
    def __init__(self, intel: Optional[IntelligenceAgent] = None):
        self.intel = intel or IntelligenceAgent()

    async def sync_news(self) -> List[Tuple[Signal, KnowledgeGraphExtraction]]:
        return []


class TacticalAgent:
    def __init__(self):
        self.adsb = ADSBScanner()
        self.aprs = APRSStreamer()
        self.sector = SectorScanner()
        self.usgs = USGSRiverGauge()
        self.netsec = NetworkAndSecurityScanner()
        self.software = LocalSoftwareScanner()
        self.rf_sweep = WidebandSDRScanner()

    async def generate_snapshot(self) -> TacticalSnapshot:
        results = await asyncio.gather(
            self.adsb.get_live_data(),
            self.sector.get_atmos_weather(),
            self.usgs.get_levels(),
            self.software.get_summary(),
            self.rf_sweep.get_snapshot_text(),
            self.netsec.get_summary(),
        )

        adsb_raw = results[0]
        weather = results[1]
        rivers = results[2]
        sw = results[3]
        rf = results[4]
        netsec = results[5]

        adsb_lines = ["### AIRSPACE SURVEILLANCE (ADS-B)"]
        aircraft_list = adsb_raw.get("aircraft", [])
        for ac in aircraft_list:
            lat, lon = ac.get("lat"), ac.get("lon")
            if lat and lon:
                flight = ac.get("flight", "UNK").strip()
                alt = ac.get("alt_baro", ac.get("alt_geom", 0))
                adsb_lines.append(
                    f"- Flight {flight} at {alt}ft (Lat: {lat}, Lon: {lon})"
                )
        if len(adsb_lines) == 1:
            adsb_lines.append("- No aircraft in detection range.")

        adsb_text = "\n".join(adsb_lines)

        raw_text = (
            f"Title: Master Tactical SITREP - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{weather['text']}\n{netsec['text']}\n{adsb_text}\n{rivers['text']}\n{sw['text']}\n{rf['text']}"
        )

        return TacticalSnapshot(
            temp_f=weather.get("temp"),
            aircraft_count=len(adsb_raw.get("aircraft", [])),
            lan_device_count=netsec["data"].get("devices", 0),
            ssh_failure_count=netsec["data"].get("ssh_fails", 0),
            internet_latency_ms=netsec["data"].get("latency"),
            rf_peaks=[
                {"freq": s["freq"], "power": s["db"]} for s in rf.get("data", [])
            ],
            rivers=rivers.get("data", []),
            software=sw.get("data", {}),
            raw_sitrep=raw_text,
        )

    async def generate_full_sitrep(self, prev: Optional[str] = None) -> str:
        snap = await self.generate_snapshot()
        return snap.raw_sitrep


class ADSBScanner:
    def __init__(self, host: str = "192.168.1.246", user: str = "pi"):
        self.host, self.user = host, user

    async def get_live_data(self) -> dict:
        cmd = [
            "ssh",
            f"{self.user}@{self.host}",
            "cat /home/pi/adsb_data/aircraft.json",
        ]
        try:
            p = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            out, err = await p.communicate()
            return (
                json.loads(out.decode())
                if p.returncode == 0
                else {"error": err.decode()}
            )
        except Exception as e:
            return {"error": str(e)}


class SectorScanner:
    def __init__(self, loc: str = "Tioga County, PA"):
        self.loc = loc

    async def get_atmos_weather(self) -> dict:
        try:
            proc = await asyncio.create_subprocess_exec(
                settings.ATMOS_BIN,
                self.loc,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            text = re.sub(r"\x1b\[[0-9;]*m", "", stdout.decode()).strip()
            temp_match = re.search(r"(\d+\.\d+)°F", text)
            return {
                "text": text,
                "temp": float(temp_match.group(1)) if temp_match else None,
            }
        except Exception:
            return {"text": "Error.", "temp": None}


class USGSRiverGauge:
    def __init__(
        self, site_codes: List[str] = ["01548500", "01531500", "01531000", "01518700"]
    ):
        self.site_codes = site_codes

    async def get_levels(self) -> dict:
        url = f"https://waterservices.usgs.gov/nwis/iv/?format=json&sites={','.join(self.site_codes)}&parameterCd=00060,00065&siteStatus=all"
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(url, timeout=30)
                if resp.status_code != 200:
                    return {"text": f"Error: {resp.status_code}", "data": []}
                data = resp.json()
                res = []
                structured = []
                for ts in data.get("value", {}).get("timeSeries", []):
                    name = ts["sourceInfo"]["siteName"]
                    val_str = ts["values"][0]["value"][0]["value"]
                    val = float(val_str)
                    unit = (
                        "ft"
                        if "height" in ts["variable"]["variableName"].lower()
                        else "cfs"
                    )
                    res.append(f"- {name}: {val} {unit}")
                    structured.append({"name": name, "value": val, "unit": unit})
                return {"text": "\n".join(sorted(list(set(res)))), "data": structured}
            except Exception as e:
                return {"text": f"USGS Error: {str(e)}", "data": []}


class WidebandSDRScanner:
    def __init__(self, host: str = "192.168.1.246", user: str = "pi"):
        self.host, self.user = host, user

    async def get_snapshot_text(self) -> dict:
        cmd = [
            "ssh",
            f"{self.user}@{self.host}",
            "python3 /home/pi/bin/radar_sdr_juggler.py",
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                return {"text": f"Wideband SDR Error: {stderr.decode()}", "data": []}
            data = json.loads(stdout.decode())
            signals = data.get("top_signals", [])
            lines = ["### FULL SPECTRUM RF SWEEP (1MHz - 1700MHz)"]
            for s in signals:
                lines.append(
                    f"- Frequency: {s['freq']:.2f} MHz | Power: {s['db']:.2f} dB"
                )
            return {"text": "\n".join(lines), "data": signals}
        except Exception as e:
            return {"text": f"Error: {str(e)}", "data": []}


class NetworkAndSecurityScanner:
    async def get_summary(self) -> dict:
        def run_sync_cmd(args):
            return subprocess.run(args, capture_output=True, text=True).stdout

        arp_output = run_sync_cmd(["sudo", "-n", "arp-scan", "-l"])
        device_count = len(
            re.findall(r"^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+", arp_output, re.MULTILINE)
        )

        ping_output = run_sync_cmd(["ping", "-c", "1", "1.1.1.1"])
        latency = re.search(r"time=([\d\.]+) ms", ping_output)
        latency_val = float(latency.group(1)) if latency else None

        auth_output = run_sync_cmd(
            ["sudo", "-n", "grep", "Failed password", "/var/log/auth.log"]
        )
        ssh_fails = len(auth_output.strip().split("\n")) if auth_output.strip() else 0

        text = f"### NETWORK & SECURITY INTEGRITY\n- **Latency:** {latency_val}ms\n- **Devices:** {device_count}\n- **SSH Fails:** {ssh_fails}"
        return {
            "text": text,
            "data": {
                "latency": latency_val,
                "devices": device_count,
                "ssh_fails": ssh_fails,
            },
        }


class LocalSoftwareScanner:
    async def get_summary(self) -> dict:
        def get_count(cmd):
            try:
                out = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True
                ).stdout.strip()
                return int(out) if out.isdigit() else 0
            except Exception:
                return 0

        apt = get_count("dpkg -l | wc -l")
        pip = get_count(f"{settings.PYTHON_BIN} -m pip list | wc -l")
        uv = get_count("uv pip list | wc -l")
        mamba = get_count("micromamba list | wc -l")

        data = {"apt": apt, "pip": pip, "uv": uv, "mamba": mamba}
        text = f"### SYSTEM SOFTWARE\n- **APT:** {apt}\n- **PIP:** {pip}\n- **UV:** {uv}\n- **MAMBA:** {mamba}"
        return {"text": text, "data": data}


class APRSStreamer:
    async def get_snapshot_text(self) -> str:
        return "APRS: Monitoring..."


class NWSAlerts:
    async def get_alerts(self) -> str:
        return "NWS: No active alerts."


class CISAFeed:
    async def get_latest_vulns(self) -> str:
        return "CISA: System nominal."


class WebIngestAgent:
    def __init__(self, intel=None):
        self.intel = intel or IntelligenceAgent()


class AudioIngestAgent:
    def __init__(self, intel=None):
        self.intel = intel or IntelligenceAgent()
