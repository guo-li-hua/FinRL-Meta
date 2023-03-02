from .task import *
from .period import *
from .report import *


_tasks = dict()
for func in celery.tasks:
    if func.startswith('app.async_task'):
        func_name = func.split('.')[-2] + '.' + func.split('.')[-1]
        _tasks[func_name] = celery.tasks[func]


def post(proc_name, data):
    """
    异步调用任务，不等待结果
    :param proc_name: 事务名称
    :param data: 参数，dict
    :return: True/False
    """
    result = _tasks[proc_name].s(data).apply_async()
    return result


def call(proc_name, data=None, sync=False):
    """
    异步或同步执行事务并等待结果
    :param proc_name: 事务名称
    :param data: 参数，dict
    :param sync: 是否同步调用
    :return: True/False
    """
    if proc_name not in _tasks:
        raise Exception('异步任务 '+proc_name+' 未找到！')
    if sync:
        return _tasks[proc_name](data)
    result = _tasks[proc_name].s(data).apply_async()
    res = result.get()
    return res
