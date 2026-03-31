with open("/home/chuck/Projects/radar/src/radar/core/ingest.py", "r") as f:
    code = f.read()

adsb_old = """class ADSBScanner:
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
            return {"error": str(e)}"""

adsb_new = """class ADSBScanner:
    def __init__(self, db_path: str = "/home/chuck/Projects/radar/radar.db"):
        self.db_path = db_path

    async def get_live_data(self) -> dict:
        import sqlite3
        import asyncio
        import json

        def fetch():
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT payload FROM mqtt_messages WHERE topic = 'camp/tioga/data/adsb' ORDER BY timestamp DESC LIMIT 1"
                )
                row = cursor.fetchone()
                conn.close()
                if row:
                    return json.loads(row[0])
                return {"aircraft": []}
            except Exception as e:
                return {"error": str(e), "aircraft": []}

        return await asyncio.to_thread(fetch)"""

rf_old = """class WidebandSDRScanner:
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
            return {"text": "\\n".join(lines), "data": signals}
        except Exception as e:
            return {"text": f"Error: {str(e)}", "data": []}"""

rf_new = """class WidebandSDRScanner:
    def __init__(self, db_path: str = "/home/chuck/Projects/radar/radar.db"):
        self.db_path = db_path

    async def get_snapshot_text(self) -> dict:
        import sqlite3
        import asyncio
        import json

        def fetch():
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT payload FROM mqtt_messages WHERE topic = 'camp/tioga/data/rf_sweep' ORDER BY timestamp DESC LIMIT 1"
                )
                row = cursor.fetchone()
                conn.close()
                
                if row:
                    data = json.loads(row[0])
                    signals = data.get("top_signals", [])
                    lines = ["### FULL SPECTRUM RF SWEEP (1MHz - 1700MHz)"]
                    for s in signals:
                        lines.append(
                            f"- Frequency: {s['freq']:.2f} MHz | Power: {s['db']:.2f} dB"
                        )
                    if len(lines) == 1:
                        lines.append("- No strong signals detected.")
                    return {"text": "\\n".join(lines), "data": signals}
                return {"text": "### FULL SPECTRUM RF SWEEP\\n- Database read error or no data.", "data": []}
            except Exception as e:
                return {"text": f"Wideband SDR Error: {str(e)}", "data": []}

        return await asyncio.to_thread(fetch)"""

code = code.replace(adsb_old, adsb_new)
code = code.replace(rf_old, rf_new)

with open("/home/chuck/Projects/radar/src/radar/core/ingest.py", "w") as f:
    f.write(code)

print("Refactor complete.")
