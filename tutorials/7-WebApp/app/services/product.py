# -*- coding:utf-8 --*--
import os, rsa, base64
import json
from datetime import datetime, date
from flask import Flask, jsonify, request, Response, session, views, redirect, render_template
import pinyin
from sqlalchemy import func, text
import requests
# from flask_login import LoginManager, login_user, login_required, current_user, logout_user
from flask import jsonify, request, render_template, blueprints
from app import app, auth, api
from app.models.base import *
from app.models import led

# from influxdb import InfluxDBClient  # Point, WritePrecision
# from influxdb.InfluxDBClient import InfluxDBClient, Point, WritePrecision
from influxdb_client import InfluxDBClient, Point, WritePrecision
# from influxdb_client.client.write_api import SYNCHRONOUS
# import paho.mqtt.client as mqtt
from flask.blueprints import Blueprint
import redis, urllib
from app.config import Config
# from aliyunsdkcore.client import AcsClient
# from aliyunsdkcore.request import CommonRequest
import pickle
from .func import *

influxdb_client = InfluxDBClient(url="http://influxdb:8086", token='tS5SITcRDdXVJpTR9wttCtGYA6Xc26udxiUodzXKdsLPB07unoO93pn6lAz_MssEjyC19cWqP7wrBHOpZcMaUQ==', org='lmshow')

# influxdb_client = InfluxDBClient(host="http://influxdb", port="8086", username="root", password="root")
product = Blueprint('product', __name__)
url = urllib.parse.urlparse(Config.REGIST_CODE_URL)
regist_code_db = 0 if url.path[1:] == '' else int(url.path[1:])
regist_code_redis = redis.Redis(host=url.hostname, port=6379 if url.port is None else url.port,
                                db=regist_code_db, password=url.password)


@product.route('/api/send_command/<imei>', methods=['POST'])
def send_command(imei):
    j = request.json
    c = led.Command()
    c.imei = imei
    p = led.Product.query.filter_by(imei=imei).first()
    c.product_id = p.id
    c.report_time = datetime.now()
    c.save()
    for s in j:
        sensor = led.Sensor.query.filter(text('product_catalog_id=:pcid and name=:name')).params(
            pcid=p.product_catalog_id, name=s['name']).first()
        cd = led.CommandDetail()
        cd.command_id = c.id
        cd.product_id = c.product_id
        cd.key = s['name']
        cd.desc = sensor.desc
        cd.value = s['value_string']
        cd.scale = sensor.scale
        cd.save()
    return api.response({'rsp': 0})


@product.route('/api/del_device/<id>', methods=['DELETE'])
def del_device(id):
    led.Product.query.get(id).delete()
    return 'True'


@product.route('/api/productcatalog/', methods=['POST'])
@product.route('/api/productcatalog/<id>', methods=['PUT'])
def add_product_catalog(id=None):
    j = request.json
    if request.method == 'POST':
        pc = led.ProductCatalog()
    else:
        pc = led.ProductCatalog.query.filter_by(id=id).first()
        s = "delete from sensor where product_catalog_id=:id"
        db.session.execute(s, {'id': id})
    for i in j:
        if i == 'sensors':
            continue
        setattr(pc, i, j[i])
    pc.save()
    for i in j['sensors']:
        s = led.Sensor(**i)
        s.product_catalog_id = pc.id
        db.session.add(s)
        db.session.commit()
    return api.response({})


@product.route('/api/_factory_register_device/', methods=['POST'])
def factory_register_device():
    # 工厂出厂注册白名单
    code = request.json['imei']
    code = base64.b64decode(code)
    path = os.path.join(os.path.dirname(__file__), 'private_key')
    with open(path, 'r') as f:
        key = rsa.PrivateKey.load_pkcs1(f.read())
    try:
        code = rsa.decrypt(code, key)
        code = bytes.decode(code)
    except Exception as ex:
        log('二维码解码错误:' + request.json['imei'])
        return api.fail('二维码解码错误')
    product = led.WhiteList.query.filter_by(imei=code).first()
    if product is not None:
        log('设备已注册:' + request.json['imei'])
        return api.fail('设备已注册')
    wl = led.WhiteList()
    wl.imei = code
    wl.save()
    return api.response('注册成功')


@product.route('/api/_device_start_register/', methods=['POST'])
def device_start_register():
    # 设备上电注册
    j = request.json
    if 'cell' not in j:
        # 不带卡
        wl = led.WhiteList.query.filter_by(imei=j['imei']).first()
        if wl is None:
            log('不在白名单设备非法上电注册' + j['imei'])
            return api.fail('failed')
        else:
            wl.register = 1
            wl.save()
            product = led.Product()
            product.imei = wl.imei
            product.save()
            return api.response('success')
    else:
        # 带卡
        wl = led.WhiteList.query.filter_by(imei=j['imei']).first()
        if wl is None:
            log('不在白名单设备非法上电注册' + j['imei'])
            return api.fail('failed')
        else:
            if wl.register != 0:
                log('白名单设备试图重新注册' + j['imei'])
                return api.fail('白名单设备试图重新注册')
            product = led.Product.query.filter_by(cell=j['cell']).first()
            if product is not None:
                log('试图注册相同手机号' + j['imei'] + '' + j['cell'])
                return api.fail('试图注册相同手机号')
            wl.register = 1
            wl.save()
            product = led.Product()
            product.imei = wl.imei
            product.cell = j['cell']
            product.save()
            return api.response('success')


@product.route('/api/register_white_list', methods=['POST'])
def register_white_list():
    usr = auth.current_user
    j = request.json
    code = request.json['imei']
    code = base64.b64decode(code)
    path = os.path.join(os.path.dirname(__file__), 'private_key')
    with open(path, 'r') as f:
        key = rsa.PrivateKey.load_pkcs1(f.read())
    try:
        code = rsa.decrypt(code, key)
        code = bytes.decode(code)
    except Exception as ex:
        return api.fail('二维码错误，请重新扫描二维码')
    wl = led.WhiteList.query.filter_by(imei=code).first()
    if wl is not None:
        return api.fail('该设备已注册白名单')
    else:
        wl = led.WhiteList()
        wl.imei = code
        wl.product_catalog_id = j['product_catalog_id']
        wl.save()
        return api.response(message='OK')


@product.route('/api/register_device', methods=['POST'])
def register_device():
    usr = auth.current_user
    j = request.json
    openid = usr.openid
    code = request.json['code']
    code = base64.b64decode(code)
    path = os.path.join(os.path.dirname(__file__), 'private_key')
    with open(path, 'r') as f:
        key = rsa.PrivateKey.load_pkcs1(f.read())
    try:
        code = rsa.decrypt(code, key)
        code = bytes.decode(code)
    except Exception as ex:
        return api.fail('二维码错误，请重新扫描二维码')
    product = led.Product.query.filter_by(imei=code).first()
    if product is None:
        wl = led.WhiteList.query.filter_by(imei=code).first()
        if wl is None:
            return api.fail('非法设备')
        product = led.Product()
        product.imei = code
        product.cell = usr.cell
        product.product_catalog_id = wl.product_catalog_id
        wx_usr = led.WxUser.query.filter_by(openid=openid).first()
        product.wx_user.append(wx_usr)
        product.save()
    else:
        if len(product.wx_user) > 0:
            return api.fail("设备已绑定")
        wx_usr = led.WxUser.query.filter_by(openid=openid).first()
        product.wx_user.append(wx_usr)
        product.save()
    return api.response(message='OK')


@product.route('/api/change_device_name', methods=['POST'])
def change_device_name():
    usr = auth.current_user
    j = request.json
    product = led.Product.query.filter_by(id=j['id']).first()
    if product is None:
        return api.fail('设备不存在')
    else:
        product.name = j['name']
        product.save()
        return api.ok


@product.route('/api/_verify_code/<cell>', methods=['GET'])
def _verify_code(cell):
    j = regist_code_redis.get(cell)
    j = pickle.loads(j)
    if j is None:
        log('校验手机号错误，redis查询失败，手机号：' + cell)
        return api.fail('校验手机号错误，redis查询失败')
    product = led.Product()
    product.imei = j['code']
    product.no = cell
    wx_user = led.WxUser.query.filter_by(openid=j['openid']).first()
    if wx_user is None:
        log('校验手机号错误，微信用户不存在，openid：' + j['openid'])
        return api.fail('校验手机号错误，微信用户不存在')
    product.wx_users.append(wx_user)
    product.save()
    log('设备注册成功，')
    return api.response('OK')


@product.route('/api/message/<int:page>/<int:page_size>/<product_id>', methods=['GET'])
def _message(page, page_size, product_id):
    offset = str((page - 1) * page_size)
    query_string = 'from(bucket: "lmshow")|> range(start: -30d)|> filter(fn: (r) => r["_measurement"] == "message")|> filter(fn: (r) => r["product_id"] == "{0}")|> filter(fn: (r) => r["_field"] == "message_detail")|> sort(columns: ["_time"], desc:true )|> limit(n: {1}, offset: {2})'.format(
        product_id, page_size, offset)
    res = influxdb_client.query_api().query(org='lmshow', query=query_string)
    res_data = list(res[0].records)
    l = list()
    for m in res_data:
        r = dict()
        r['message_detail'] = json.loads(m.get_value())
        l.append(r)
    return api.response(l)


@product.route('/api/get_last_gps_data/<product_id>', methods=['GET'])
def get_last_gps_data(product_id):
    res = influxdb_client.query('select * from gps_data where product_id=$id order by time desc limit 1 offset 0;',
                                {'id': product_id})
    res_data = list(res.get_points())
    for m in res_data:
        m['message_detail'] = json.loads(m['message_detail'])
    return api.response(res_data)


@product.route('/upgrade', methods=['POST'])
def upgrade():
    if len(request.files) == 0:
        return "haven't upgrade file!"
    return 'ok'


###########################################################################
# User
###########################################################################
@product.route('/api/user/list')
def user_list():
    return led.json(led.User.query.all())


@product.route('/user/<id>', methods=['GET', 'DELETE'])
def user(id):
    if request.method == 'GET':
        usr = led.User.query.get(id)
        if usr is None:
            return 'false'
        else:
            # usr.access = eval(usr.access)
            return jsonify(usr.to_dict())
    elif request.method == 'DELETE':
        led.User.query.get(id).delete()
        return 'true'


@product.route('/user', methods=['POST'])
def user_create():
    j = request.json
    usr = led.User(**j)
    db.session.add(usr)
    db.session.commit()
    return 'true'


@product.route('/change_password', methods=['POST'])
def change_password():
    j = request.json
    usr = led.User.query.filter_by(user_id=j['user_id']).first()
    usr.password = j['password']
    usr.commit()
    session.clear()
    return 'true'


@product.route('/api/device_code/<code>')
def device_code(code):
    dev = led.Device.query.filter_by(code=code).first()
    if dev is None:
        return jsonify({})
    else:
        return jsonify(dev.to_dict())


@product.route('/api/name_value/<int:type_id>/<name>', methods=['GET'])
def get_sys_dict(type_id, name):
    obj = led.SysDict.query.filter_by(type_id=type_id, name=name).first()
    if obj is None:
        return jsonify({})
    else:
        return jsonify(obj.to_dict())


@product.route('/api/value_name/<int:type_id>/<value>', methods=['GET'])
def value_name(type_id, value):
    obj = led.SysDict.query.filter_by(type_id=type_id, value=value).first()
    if obj is None:
        return jsonify({})
    else:
        return jsonify(obj.to_dict())


@product.route('/api/upload', methods=['POST'])
def upload():
    if not request.files:
        return jsonify({})
    for f in request.files:
        attach = led.Attach()
        attach.save(request.files[f])
        attach = led.Attach.query.get(attach.id)
        j = attach.to_dict()
        return jsonify(j)


@product.route('/api/upload/<int:id>')
def upload_url(id):
    attach = led.Attach.query.get(id)
    if os.name != 'nt':
        return redirect(os.path.join('/attach/', attach.path, attach.filename))
    with open(os.path.join('upload', attach.path, attach.filename), 'r') as f:
        return f.read()
