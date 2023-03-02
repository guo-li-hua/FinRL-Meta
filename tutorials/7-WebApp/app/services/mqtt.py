from app.models.base import *
from app.models import led
from influxdb import InfluxDBClient
import paho.mqtt.client as mqtt
import json
from datetime import datetime
import hashlib, base64


influxdb_client = InfluxDBClient('influxdb', 8086, 'admin', 'adminadmin', 'led')
mqtt_broke = 'post-cn-nif1z2db71s.mqtt.aliyuncs.com'
mqtt_username = 'Signature|LTAI4GFKi8yeQgwpxK4vs2V6|post-cn-nif1z2db71s'
mqtt_password = 'hUILi4Us9Px37wlgduwRuKsz8s8='
mqtt_client_id = 'GID_LED@@@server'


def on_report(message):
    p = led.Product.query.filter_by(imei=message['device_id']).first()
    if p is None:
        return False
    d = message['data']
    ss = led.Sensor.query.filter_by(product_catalog_id=p.product_catalog_id).all()
    message_details = []
    for s in ss:
        message_detail = {}
        message_detail['key'] = s.name
        message_detail['desc'] = s.desc
        message_detail['scale'] = s.scale
        try:
            message_detail['value'] = d[s.name]
        except Exception as e:
            pass
        message_details.append(message_detail)
    #influxdb
    point = [
        {
            "measurement": "message",
            "tags": {
                "product_id": p.id,
                "imei": message['device_id']
            },
            "fields": {
                # "report_time": str(datetime.strptime(j['iotEventTime'],'%Y-%m-%d %H:%M:%S')),
                "message_detail": json.dumps(message_details)
            }
        }
    ]
    res = influxdb_client.write_points(point)
    return True

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("led_control/message")


# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    d = msg.payload.decode()
    m = json.loads(d)
    if m['type'] == 'REPORT':
        on_report(m)
    else:
        pass
    print(msg.topic + " " + str(msg.payload))


client = mqtt.Client(mqtt_client_id)
client.username_pw_set(mqtt_username,mqtt_password)
client.on_connect = on_connect
client.on_message = on_message

client.connect(mqtt_broke, 1883, 60)
client.loop_forever()
