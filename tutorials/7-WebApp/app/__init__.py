# -*- coding:utf-8 --*--
import logging
from flask import Flask
from flask_compress import Compress
from app.extensions import *
import app.models
import app.services
import app.async_task


def create_app():
    from .config import Config
    app = Flask(__name__)
    app.config.from_object(Config)
    Compress(app)

    services.init_app(app)
    extensions.init_app(app)
    if not Config.DEBUG:
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
    return app
