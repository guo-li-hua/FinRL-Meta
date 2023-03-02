from datetime import datetime, timedelta
from time import sleep
import pandas as pd
from pandas.tseries.offsets import Day
from copy import deepcopy
import re
from io import BytesIO


def dtype_coltype(dtype):
    """pandas dtypes to base table col_type"""
    coltype = list()
    for item in dtype:
        coltype.append({'name':item, 'type':dtype[item].name, 'index':0})
    return coltype


def str2month(s):
    if len(s)<10:
        s = s + '-01'
    begin = str2datetime(s)
    begin = datetime(begin.year, begin.month, 1)
    end = begin + timedelta(days=31)
    end = datetime(end.year, end.month, 1)
    return [begin, end]


def str2datetime(s):
    if s is None:
        return s
    d = map(lambda x: int(x), re.findall('\d+', s))
    d = list(d)
    if len(d) < 3:
        return datetime(1,1,1)
    d = datetime(*d)
    if s.endswith('Z'):
        d += timedelta(hours=8)
    return d


def find_obj(l, key, value):
    for i in l:
        if i[key] == value:
            return i[key]
    return None


def parse_param(param_def, kwargs):
    """处理参数类型，计算表达式，获得参数dict"""
    params = dict()
    for param in param_def:
        if param['var'].startswith('@'):
            param['var'] = param['var'][1:]
        if param['var'] in kwargs:
            param['val'] = kwargs[param['var']]
        if 'date' in param['type'] and isinstance(param['val'], str):
            param['val'] = str2datetime(param['val'])
        params[param['var']] = param['val']

    # eval expression
    _globals = deepcopy(params)
    _globals['DAYS'] = Day
    for param in param_def:
        if param['type'] != 'expression' or not isinstance(param['val'], str):
            continue
        if re.search('[\+\-\*/]', param['val']) is None:
            continue
        params[param['var']]= eval(param['val'], _globals)
    return params


def excel_append(filename, df):
    stream = BytesIO()
    with open(filename, 'rb') as f:
        stream.write(f.read())
    wr = pd.ExcelWriter(stream, engine='openpyxl', mode='a')
    df.to_excel(wr, sheet_name='data')
    wr.save()
    wr.close()
    stream.seek(0)
    # s = stream.read()
    return stream


def wait_result(res, wait_time=5):
    """在等待时间内读异步任务结果
    :param res:AsyncResult
    :param wait_time:等待时间，默认5秒
    :return:DataFrame or None
    """
    for i in range(wait_time*10):
        if res.state == 'PENDING':
            sleep(0.1)
        else:
            break
    if res.state == 'SUCCESS':
        data = res.get()
        return data
    else:
        return None


def b64(n):
    table = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-_'
    result = []
    temp = n
    if 0 == temp:
      result.append('0')
    else:
      while 0 < temp:
        result.append(table[temp % 64])
        temp = temp//64
    return ''.join([x for x in reversed(result)])


def connect_redis(url):
    import redis
    import urllib
    url = urllib.parse.urlparse(url)
    if url.scheme != 'redis':
        raise Exception('need url: <redis://ip:port/db>')
    db = 0 if url.path[1:] == '' else int(url.path[1:])
    return redis.Redis(host=url.hostname, port=6379 if url.port is None else url.port, db=db, password=url.password)


def check_code_ok(verify, code):
    try:
        timestamp = (datetime.now() + timedelta(weeks=int(code), hours=int(code)) - datetime(1970, 1, 1, 8)).total_seconds()
        diff = abs(timestamp - int(verify, 16))
    except Exception as ex:
        return False
    if 60<diff<600:
        raise Exception('验证码已失效')
    elif diff > 600:
        return False
    return True
