# -*- coding:utf-8 --*--
import json
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
from app.models import led

def dtype_coltype(dtype):
    """pandas dtypes to base table col_type"""
    coltype = list()
    for item in dtype:
        coltype.append({'name':item, 'type':dtype[item].name, 'index':0})
    return coltype


def send_sms(phone_num,code):
    client = AcsClient('LTAIqtY8AWK41cPq', 'nR0WuPXMEeYzQIymXvP6EIWetbCoxb', 'default')
    request = CommonRequest()
    request.set_accept_format('json')
    request.set_domain('dysmsapi.aliyuncs.com')
    request.set_method('POST')
    request.set_protocol_type('https')  # https | http
    request.set_version('2017-05-25')
    request.set_action_name('SendSms')
    request.add_query_param('RegionId', "default")
    request.add_query_param('PhoneNumbers', phone_num)
    request.add_query_param('SignName', "凌硕科技")
    request.add_query_param('TemplateCode', "SMS_173247376")
    params = {'code':code}
    js = json.dumps(params)
    request.add_query_param('TemplateParam', js)
    response = client.do_action_with_exception(request)
    rs = json.loads(response)
    return rs


def log(message):
    l = led.Log()
    l.message = message
    l.save()
    return True
