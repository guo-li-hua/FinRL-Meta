# -*- coding:utf-8 --*--
import sys
import os
import yaml


working_dir = os.path.abspath(sys.argv[0])
working_dir = os.path.split(working_dir)[0]
os.chdir(working_dir)
os.environ['WORKING_DIR'] = working_dir


class Config(object):
    EXT = dict()
    # ############### flask settings ##################
    DEBUG = True
    SECRET_KEY = ',sdf34%$sd'
    JSON_AS_ASCII = False
    JSON_SORT_KEYS = False
    WORKING_DIR = working_dir

    # ############## lings settings ##################
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(working_dir, 'forecast.db')
    SQLALCHEMY_BINDS = {}
    # use key-value pair config extra db connection string
    # db.extra_db_uri = {
    #     'index': 'sqlite:///' + os.path.join(working_dir, 'index.db'),
    #     'other': 'mssql+pymssql://' }

    # None is memory auth, or redis://:pass@ip/1 set redis auth
    LINGS_AUTH_URL = 'redis://redis-1/2'  # None
    LINGS_AUTH_EXPIRE = 24*60*30               # 登录失效时间（分钟）
    LINGS_API_PREFIX = '/api/'

    # ############### celery settings ##################
    CELERY_TIMEZONE	= 'Asia/Shanghai'
    BROKER_URL = 'redis://redis-1:6379/0'
    CELERY_RESULT_BACKEND = 'redis://redis-1:6379/0'
    CELERY_TASK_SERIALIZER = 'pickle'
    CELERY_RESULT_SERIALIZER = 'pickle'
    CELERY_ACCEPT_CONTENT = ['pickle','json']

    # CELERYD_TASK_TIME_LIMIT = 300
    CELERY_TASK_RESULT_EXPIRES = 24 * 3600
    # worker的并发数，默认是服务器的内核数目, 也是命令行 - c参数指定的数目
    # CELERYD_CONCURRENCY = 4
    # CELERY_MESSAGE_COMPRESSION = 'zlib'
    # 每个worker执行了多少任务就会死掉，默认是无限的
    # CELERYD_MAX_TASKS_PER_CHILD = 40

    ES_URL = 'http://192.168.1.3:9200'

    # ###############       SMS        ##################
    SMS_VERIFY_DB = 'redis://redis-1:6379/1'

    def __init__(self):
        global working_dir
        path = 'config.yml'
        # start from celery need --workdir
        if '--workdir' in sys.argv:
            working_dir = sys.argv[sys.argv.index('--workdir') + 1]
            path = os.path.join(working_dir, path)

        with open(path, encoding='utf-8') as f:
            new_config = yaml.load(f, yaml.Loader)
            for key in new_config:
                if isinstance(new_config[key], str) and '${WORKING_DIR}' in new_config[key]:
                    new_config[key] = new_config[key].replace('${WORKING_DIR}', working_dir)
                setattr(Config, key, new_config[key])

        fs = os.listdir(os.path.join(working_dir, 'app/config'))
        for f in fs:
            if f.endswith('.yml'):
                with open(os.path.join(working_dir, 'app/config', f), encoding='utf-8') as fp:
                    setattr(Config, f[:-4], yaml.load(fp, yaml.Loader))


app_settings = Config()


