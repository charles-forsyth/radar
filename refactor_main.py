with open("/home/chuck/Projects/radar/src/radar/main.py", "r") as f:
    code = f.read()

target = """                snapshot = await agent.generate_snapshot()
                db_snapshot = TacticalSnapshotModel(
                    temp_f=snapshot.temp_f,
                    aircraft_count=snapshot.aircraft_count,
                    mapped_aircraft_count=snapshot.mapped_aircraft_count,
                    lan_device_count=snapshot.lan_device_count,
                    ssh_failure_count=snapshot.ssh_failure_count,
                    internet_latency_ms=snapshot.internet_latency_ms,
                    rf_peaks=snapshot.rf_peaks,
                    rivers=snapshot.rivers,
                    software=snapshot.software,
                    raw_sitrep=snapshot.raw_sitrep,
                )"""

replacement = """                snapshot = await agent.generate_snapshot()
                db_snapshot = TacticalSnapshotModel(
                    temp_f=snapshot.temp_f,
                    aircraft_count=snapshot.aircraft_count,
                    mapped_aircraft_count=snapshot.mapped_aircraft_count,
                    lan_device_count=snapshot.lan_device_count,
                    ssh_failure_count=snapshot.ssh_failure_count,
                    internet_latency_ms=snapshot.internet_latency_ms,
                    rf_peaks=snapshot.rf_peaks,
                    rivers=snapshot.rivers,
                    software=snapshot.software,
                    raw_sitrep=snapshot.raw_sitrep,
                )
                
                # --- AURA MQTT DECENTRALIZED BUS INJECTION ---
                from radar.mqtt_client import RadarMQTTPublisher
                mqtt_pub = RadarMQTTPublisher()
                # Create a clean dict for the broker (excluding raw_sitrep string to save bandwidth)
                osint_payload = {
                    "temp_f": snapshot.temp_f,
                    "lan_device_count": snapshot.lan_device_count,
                    "ssh_failure_count": snapshot.ssh_failure_count,
                    "internet_latency_ms": snapshot.internet_latency_ms,
                    "rivers": snapshot.rivers,
                    "software": snapshot.software
                }
                mqtt_pub.publish_snapshot(osint_payload)
                # ---------------------------------------------"""

code = code.replace(target, replacement)

with open("/home/chuck/Projects/radar/src/radar/main.py", "w") as f:
    f.write(code)

print("Main refactor complete.")
