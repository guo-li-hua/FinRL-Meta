# -*- coding:utf-8 --*--
import sys
import os
from lings import ORM
from app.extensions import db

working_dir = os.path.abspath(sys.argv[0])
working_dir = os.path.split(working_dir)[0]
# db = ORM('mysql+pymysql://root:admin@localhost:3306/zliot?charset=utf8&autocommit=true')
# db = ORM('mysql+pymysql://zliot:zliot.123@www.mqtt.lmshow.net:3306/zliot?charset=utf8&autocommit=true')


# use key-value pair config extra db connection string
# db.extra_db_uri = {
#     'index': 'sqlite:///' + os.path.join(working_dir, 'index.db'),
#     'other': 'mssql+pymssql://'
# }

from .base import *
from .led import *
