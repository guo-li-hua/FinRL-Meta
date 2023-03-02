# -*- coding:utf-8 --*--
from flask import Flask, request, render_template
from flask.blueprints import Blueprint
from app.models import led
import requests
from app.extensions import auth, api
from app.wx.wxcrypt import WXBizDataCrypt
import redis, urllib
from app.config import Config

# from influxdb import InfluxDBClient  Point, WritePrecision

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import json
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException
from aliyunsdkcore.auth.credentials import AccessKeyCredential
from aliyunsdkcore.auth.credentials import StsTokenCredential
# from aliyunsdkonsmqtt.request.v20200420.RegisterDeviceCredentialRequest import RegisterDeviceCredentialRequest
import paho.mqtt.client as mqtt

# change working dir
# working_dir = os.path.abspath(sys.argv[0])
# working_dir = os.path.split(working_dir)[0]
# os.chdir(working_dir)
#
# # create flask app
# app = Flask(__name__)
# app.config['SECRET_KEY'] = ',sdf34%$sd'
# app.config['JSON_AS_ASCII'] = False
# app.config['JSON_SORT_KEYS'] = False
# Compress(app)
#
# create restful api object from model
# api = Restful(app, Model, debug=True)
# create auth object
# auth = Auth(app, url='redis://www.mqtt.lmshow.net:6379/0',debug=True)


main = Blueprint('main', __name__)

url = urllib.parse.urlparse(Config.WX_OPENID_URL)
openid_db = 0 if url.path[1:] == '' else int(url.path[1:])
openid_redis = redis.Redis(host=url.hostname, port=6379 if url.port is None else url.port,
                           db=openid_db, password=url.password)
influxdb_client = InfluxDBClient(url="http://influxdb:8086", token='tS5SITcRDdXVJpTR9wttCtGYA6Xc26udxiUodzXKdsLPB07unoO93pn6lAz_MssEjyC19cWqP7wrBHOpZcMaUQ==', org='lmshow')
mqtt_instance_id = 'post-cn-nif1z2db71s'
access_key_id = 'LTAI4GFKi8yeQgwpxK4vs2V6'
access_key_secret = 'R9eZaQJKw6ULxoSE4xOsiN8FUlAyz7'
# mqtt_broke = 'post-cn-nif1z2db71s.mqtt.aliyuncs.com'
mqtt_broke = 'mqtt.mqtt.lmshow.net'
# 如果连接阿里云MQTT服务器，需要计算获得username，password
# mqtt_username = 'Signature|LTAI4GFKi8yeQgwpxK4vs2V6|post-cn-nif1z2db71s'
# mqtt_password = 'hUILi4Us9Px37wlgduwRuKsz8s8='
import random
mqtt_client_id = 'server' + str(random.randint(100000,900000))


def on_disconnect(client, userdata, rc):
    if rc != 0:
        print("Unexpected disconnection.")
    try:
        res = client.reconnect()
    except Exception:
        pass
    return


def on_connect(client, userdata, flags, rc):
    print("connected")


def cal_username_password(device_id):
    # """得到连接mqtt的username，password"""
    # credentials = AccessKeyCredential(access_key_id, access_key_secret)
    # # use STS Token
    # # credentials = StsTokenCredential('<your-access-key-id>', '<your-access-key-secret>', '<your-sts-token>')
    # client = AcsClient(region_id='cn-qingdao', credential=credentials)
    # request1 = RegisterDeviceCredentialRequest()
    # request1.set_accept_format('json')
    # client_id = 'GID_LED@@@' + device_id
    # request1.set_ClientId(client_id)
    # request1.set_InstanceId("post-cn-nif1z2db71s")
    # response = client.do_action_with_exception(request1)
    # response_dict = json.loads(response)
    # import base64, hashlib, hmac
    res = dict()
    res['username'] = 'DeviceCredential|' + response_dict['DeviceCredential']['DeviceAccessKeyId'] +'|post-cn-nif1z2db71s'
    h = hmac.new(response_dict['DeviceCredential']['DeviceAccessKeySecret'].encode('UTF-8'),client_id.encode('UTF-8'),hashlib.sha1)
    res['password'] = base64.b64encode(h.digest()).decode()
    return res


client = mqtt.Client('GID_LED@@@'+mqtt_client_id, False)
# 如果连接阿里云MQTT服务器，需要计算获得username，password
# up = cal_username_password(mqtt_client_id)
# client.username_pw_set(up['username'],up['password'])
client.username_pw_set('server','server123')
client.on_disconnect = on_disconnect
client.on_connect = on_connect
res = client.connect(mqtt_broke, 1883, 60)
client.loop_start()

@main.route('/api/_init')
def _init():
    """初始化数据库"""
    led.db.create_all()
    return 'OK'


@main.route('/api/_funcs')
def funcs():
    """读取后台服务清单"""
    obj = api.funcs
    return render_template('api.html', funcs=obj)


@main.route('/api/_login', methods=['POST'])
def login():
    """用户登录"""
    j = request.json
    usr = led.User.query.filter_by(user_id=j['user_id']).first()
    if usr is None:
        return api.fail(u'用户不存在')
    if 'openid' in j:
        usr.openid = j['openid']
        usr.save()
    if usr.password == j['password']:
        token_str = auth.login(usr)
        return api.response({'token': token_str})
    else:
        return api.fail(u'用户密码不正确')


@main.route('/api/_register_wx_user', methods=['POST'])
def _register_wx_user():
    """微信用户注册"""
    j = request.json
    session_key = bytes.decode(openid_redis.get(j['openid']))
    pc = WXBizDataCrypt(Config.APPID, session_key)
    des_data = pc.decrypt(j['data'], j['iv'])
    wx_user = led.WxUser.query.filter_by(openid=j['openid']).first()
    if wx_user is not None:
        return api.fail('用户已经存在')
    wx_user = led.WxUser()
    wx_user.openid = j['openid']
    wx_user.cell = des_data['purePhoneNumber']
    wx_user.save()
    return api.response('OK')


@main.route('/api/_get_username_password', methods=['POST'])
def _get_username_password():
    """得到连接mqtt的username，password"""
    j = request.json
    res = cal_username_password(j['device_id'])
    return res


@main.route('/api/_on_report', methods=['POST'])
def _on_report():
    """设备上报数据"""
    message = request.json
    p = led.Product.query.filter_by(imei=message['device_id']).first()
    if p is None:
        return api.fail()
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
    write_api = influxdb_client.write_api(write_options=SYNCHRONOUS)
    point = Point('message') \
        .tag("product_id",p.id) \
        .tag("imei",message['device_id']) \
        .field("message_detail", json.dumps(message_details))
    res = write_api.write(bucket='lmshow', org='lmshow', record=point)
    return api.ok


@main.route('/api/_push_gps', methods=['POST'])
def _push_gps():
    """设备上报gps数据"""
    message = request.json
    p = led.Product.query.filter_by(imei=message['device_id']).first()
    if p is None:
        return api.fail()
    gps_data = dict()
    gps_data['latitude'] = message['latitude']
    gps_data['longitude'] = message['longitude']
    #influxdb
    point = [
        {
            "measurement": "gps_data",
            "tags": {
                "product_id": p.id,
                "imei": message['device_id']
            },
            "fields": {
                # "report_time": str(datetime.strptime(j['iotEventTime'],'%Y-%m-%d %H:%M:%S')),
                "message_detail": json.dumps(gps_data)
            }
        }
    ]
    res = influxdb_client.write_points(point)
    return api.ok


@main.route('/api/_login_openid/<openid>', methods=['GET'])
def login_openid(openid):
    """微信用户登录"""
    usr = led.WxUser.query.filter_by(openid=openid).first()
    if usr is None:
        return api.fail(u'用户不存在')
    token_str = auth.login(usr)
    return api.response(usr.to_dict())


@main.route('/api/_get_openid/<appid>/<secret>/<code>', methods=['GET'])
def get_openid(appid,secret,code):
    """得到微信用户的openid"""
    url = 'https://api.weixin.qq.com/sns/jscode2session?appid={0}&secret={1}&js_code={2}&grant_type=authorization_code'.format(appid,secret,code)
    rsp = requests.get(url).json()
    openid_redis.set(rsp['openid'],rsp['session_key'])
    return api.response(rsp)


@main.route('/api/token/<token_str>', methods=['GET'])
def token(token_str):
    """根据token获得用户信息"""
    # resp = urlopen('http://47.105.210.204:6010/api/token/{0}'.format(token_str))
    resp = auth.get_user(token_str)
    return api.response(resp.to_dict())


@main.route('/api/wx_send_command', methods=['POST'])
def wx_send_command():
    """处理微信下发的开关指令"""
    usr = auth.current_user
    j = request.json
    p = led.Product.query.filter_by(imei=j['device_id']).first()
    if p is None:
        return api.fail()
    if p.wx_user[0].openid != usr.openid:
        return api.fail()
    topic = 'led_control/'+j['device_id']
    j['type'] = "switch"
    res = client.publish(topic=topic,payload=json.dumps(j),qos=1)
    return api.ok


@main.route('/api/scheduler_config_save', methods=['POST'])
def scheduler_config_save():
    usr = auth.current_user
    j = request.json
    product = led.Product.query.filter_by(imei=j['imei']).first()
    if product.wx_user[0].openid != usr.openid:
        return api.fail()
    if product is not None:
        led.ScheduleConfig.query.filter_by(imei=j['imei']).delete()
    config = led.ScheduleConfig()
    config.imei = j['imei']
    config.product_id = j['product_id']
    config.config = j['config']
    config.save()
    topic = 'led_control/'+j['imei']
    command = dict()
    command['type'] = "scheduler"
    res = client.publish(topic=topic,payload=json.dumps(command),qos=1)
    return api.ok


@main.route('/api/_get_scheduler_config/<device_id>', methods=['GET'])
def _get_scheduler_config(device_id):
    config = led.ScheduleConfig.query.filter_by(imei=device_id).first()
    if config is None:
        config = dict()
    else:
        config = config.config
    return api.response(config)


@main.route('/api/unbound', methods=['POST'])
def unbound():
    """解除设备绑定"""
    usr = auth.current_user
    j = request.json
    p = led.Product.query.filter_by(id=j['device_id']).first()
    if p is None:
        return api.fail()
    if p.wx_user[0].openid != usr.openid:
        return api.fail()
    p.wx_user = list()
    p.save()
    return api.ok


