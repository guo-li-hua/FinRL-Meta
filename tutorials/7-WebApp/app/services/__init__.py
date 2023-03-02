# -*- coding:utf-8 --*--
from .trigger import *
from .product import *
from .main import *

def init_app(app):
    app.register_blueprint(main)
    app.register_blueprint(product)
