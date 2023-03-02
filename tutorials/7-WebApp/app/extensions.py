# -*- coding:utf-8 --*--
from lings import ORM, Auth, Restful
from celery import Celery,Task

db = ORM()
auth = Auth()
api = Restful(auth)
celery = Celery(__name__)


def init_celery(flask_app):
    global celery

    # celery = Celery(__name__)
    celery.config_from_object(flask_app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with flask_app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery


def init_app(app):
    db.init_app(app)
    auth.init_app(app)
    from app import models
    api.init_app(app, models)
    global celery
    celery = init_celery(app)
