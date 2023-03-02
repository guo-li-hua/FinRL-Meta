# -*- coding:utf-8 --*--
from app.extensions import db
from .base import *
from datetime import datetime
wx_user_product = db.Table('wx_user_product',
                        db.Column('wx_user_id', db.Integer, db.ForeignKey('wx_user.id')),
                        db.Column('product_id', db.Integer, db.ForeignKey('product.id')))


class WxUser(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    openid = db.Column(db.String(50))
    nick_name = db.Column(db.String(50))
    cell = db.Column(db.String(20))
    products = db.relationship('Product', secondary=wx_user_product, lazy='joined')


class DictType(db.Model):
    """数据字典表"""
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    type = db.Column(db.String(40))             # 类型
    name = db.Column(db.String(200))
    dicts = db.relationship('SysDict', lazy='joined')


class SysDict(db.Model):
    """数字字典表明细"""
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    type_id = db.Column(db.Integer, db.ForeignKey('dict_type.id'))
    name = db.Column(db.String(200))
    value = db.Column(db.String(200))


class ProductCatalog(db.Model):
    """产品目录"""
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    code = db.Column(db.String(30))
    name = db.Column(db.String(30))
    sensors = db.relationship('Sensor', lazy='joined')


class Sensor(db.Model):
    """传感器"""
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(50))
    desc = db.Column(db.String(50))
    scale = db.Column(db.String(20))
    type = db.Column(db.String(10))
    product_catalog_id = db.Column(db.Integer, db.ForeignKey('product_catalog.id', ondelete='CASCADE'))


class Product(db.Model):
    """设备"""
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    imei = db.Column(db.String(30))
    cell = db.Column(db.String(15)) #电话号码
    name = db.Column(db.String(50))
    product_catalog_id = db.Column(db.Integer, db.ForeignKey('product_catalog.id', ondelete='CASCADE'))
    product_catalog = db.relationship('ProductCatalog',lazy='joined')
    schedule_config = db.relationship('ScheduleConfig', lazy='joined')
    # wx_user_id = db.Column(db.Integer, db.ForeignKey('wx_user.id'))
    wx_user = db.relationship('WxUser',secondary=wx_user_product, lazy='joined')
    location = db.Column(db.String(255))
    status = db.Column(db.Integer) #0:正常，1：等待认证，2：等待激活
    # messages = db.relationship('Message', lazy='dynamic')


class WhiteList(db.Model):
    """设备"""
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    imei = db.Column(db.String(30))
    register = db.Column(db.Integer, default=0) #0:未注册，1：已注册
    create_time = db.Column(db.DateTime, default=datetime.now)
    product_catalog_id = db.Column(db.Integer)

# class Message(db.Model):
#     """上报消息"""
#     id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     product_id = db.Column(db.Integer,db.ForeignKey('product.id',ondelete='CASCADE'))
#     imei = db.Column(db.String(30))
#     report_time = db.Column(db.DateTime)
#     receive_time = db.Column(db.DateTime)
#     battery = db.Column(db.Integer)
#     signal = db.Column(db.Integer)
#     message_detail = db.relationship('MessageDetail', lazy='joined')
#
#     __mapper_args__ = {
#         "order_by":receive_time.desc()
#     }


# class MessageDetail(db.Model):
#     """上报消息数据细节"""
#     id = db.Column(db.Integer, primary_key=True, autoincrement=True)
#     product_id = db.Column(db.Integer)
#     Message_id = db.Column(db.Integer,db.ForeignKey('message.id',ondelete='CASCADE'))
#     key = db.Column(db.String(20))
#     value = db.Column(db.String(20))
#     desc = db.Column(db.String(20))
#     scale = db.Column(db.String(10))


class Command(db.Model):
    """下发指令"""
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_id = db.Column(db.Integer,db.ForeignKey('product.id',ondelete='CASCADE'))
    imei = db.Column(db.String(30))
    report_time = db.Column(db.DateTime)
    receive_time = db.Column(db.DateTime)
    response = db.Column(db.String(10))
    response_detail = db.Column(db.String(50))
    command_detail = db.relationship('CommandDetail', lazy='joined')

    __mapper_args__ = {
        # "order_by":receive_time.desc()
    }


class CommandDetail(db.Model):
    """下发指令返回数据"""
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_id = db.Column(db.Integer)
    command_id = db.Column(db.Integer,db.ForeignKey('command.id',ondelete='CASCADE'))
    key = db.Column(db.String(20))
    value = db.Column(db.String(20))
    desc = db.Column(db.String(20))
    scale = db.Column(db.String(10))


class Log(db.Model):
    """日志"""
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    message = db.Column(db.String(100))
    create_time = db.Column(db.DateTime, default=datetime.now)

    __mapper_args__ = {
        # "order_by":create_time.desc()
    }


class ScheduleConfig(db.Model):
    """定时开关配置"""
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_id = db.Column(db.Integer,db.ForeignKey('product.id',ondelete='CASCADE'))
    imei = db.Column(db.String(30))
    config = db.Column(db.PickleType)


class GpsData(db.Model):
    """定时开关配置"""
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_id = db.Column(db.Integer,db.ForeignKey('product.id',ondelete='CASCADE'))
    imei = db.Column(db.String(30))
