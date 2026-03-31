import paho.mqtt.client as mqtt
import json
import logging

logger = logging.getLogger(__name__)


class RadarMQTTPublisher:
    def __init__(self, host="192.168.1.246", port=1883, user="camp", password="tioga"):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.topic = "camp/tioga/data/osint"

    def publish_snapshot(self, snapshot_dict: dict):
        try:
            client = mqtt.Client()
            client.username_pw_set(self.user, self.password)
            client.connect(self.host, self.port, 60)
            client.loop_start()

            payload = json.dumps(snapshot_dict, default=str)
            info = client.publish(self.topic, payload, qos=1)
            info.wait_for_publish()

            client.loop_stop()
            client.disconnect()
            logger.info(f"Successfully published OSINT snapshot to {self.topic}")
        except Exception as e:
            logger.error(f"Failed to publish to MQTT broker: {e}")
