#coding=utf-8
import random
import json
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
import datetime


client = AcsClient('???', '???', 'cn-hangzhou')


def send_sms(**kwargs):
    request = CommonRequest()
    request.set_accept_format('json')
    request.set_domain('dysmsapi.aliyuncs.com')
    request.set_method('POST')
    request.set_protocol_type('https') # https | http
    request.set_version('2017-05-25')
    request.set_action_name('SendSms')

    for key in kwargs:
        request.add_query_param(key, kwargs[key])

    response = client.do_action_with_exception(request)
    response = json.loads(response,encoding='utf-8')
    print(response['Message'])
    return response


def send_notify(phone, metal_base_name, status, number, date_string):
    """
    发送提示消息
    :param phone: 电话号码，多号码逗号隔开
    :param metal_base_name: 矿库名称
    :param status: 状态
    :return:
    """
    status = ''
    resp = send_sms(PhoneNumbers=phone, SignName='签名', TemplateCode='SMS_XXX',
                    TemplateParam=json.dumps(dict(name=metal_base_name, status=status, number=number, date=date_string)))
    return resp['Message']
