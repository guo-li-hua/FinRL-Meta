# -*- coding:utf-8 --*--
from datetime import datetime, timedelta
from app.extensions import celery
from celery.schedules import crontab
from app import db
from sqlalchemy import text
import pandas as pd
from app.public import str2month
from app.config import app_settings
from celery import signature


@celery.task
def parse_ticket_expire():
    print('parse ticket expire ...')
    res = db.session.execute(text(app_settings.sql['update_ticket_expire']))
    db.session.commit()
    print('...parse ticket expire done.')


@celery.task
def parse_sleep_days():
    print('calculate sleep days ...')
    res = db.session.execute(text(app_settings.sql['calc_sleep_days']))
    db.session.commit()
    print('...calculate sleep days done.')


@celery.task
def parse_info_pub():
    print('parse info pub ...')
    res = db.session.execute(text(app_settings.sql['pub_material']))
    db.session.commit()
    print('...parse info pub done.')


@celery.task
def update_station_customer_level(station_id):
    yesterday = datetime.now() - timedelta(days=1)
    begin, end = str2month(str(yesterday))
    print('update station customer level of ' + str(station_id))
    customer_df = pd.read_sql(text(app_settings.sql['month_customer_fuel_sum']), db.engine,
                              params=dict(begin=begin, end=end, station_id=station_id))
    customer_df['qiyou'] = customer_df['all_fuel']-customer_df['caiyou']-customer_df['lng']
    customer_df['level_id'] = 1
    customer_df['level_name'] = '白银会员'
    customer_df['level_rate'] = 1
    setting = marketing.StationSettings.query.filter_by(station_id=station_id, type='level_point_setting').first()
    if setting is None:
        return
    for i in range(3):
        for rule_key in setting.settings:
            rule = pd.Series(setting.settings[rule_key][i])
            # 比较月加油次数
            # compare = customer_df['times'] >= rule['monthtimes']
            # customer_df['level_id'] = customer_df['level_id'].mask(compare, rule['id'])
            # customer_df['level_name'] = customer_df['level_name'].mask(compare, rule['name'])
            # customer_df['level_rate'] = customer_df['level_rate'].mask(compare, rule['times'])
            # 比较加油金额
            compare = customer_df[rule_key] >= rule['monthtotal']
            customer_df['level_id'] = customer_df['level_id'].mask(compare, rule['id'])
            customer_df['level_name'] = customer_df['level_name'].mask(compare, rule['name'])
            customer_df['level_rate'] = customer_df['level_rate'].mask(compare, rule['times'])
    customer = customer_df[customer_df['old_level'] != customer_df['level_id']]
    if customer.empty:
        return
    db.session.execute(text(app_settings.sql['update_customer_level']), params=customer.to_dict(orient='records'))
    db.session.commit()
    print('update station ' + str(station_id) + ' done.')


@celery.task
def update_customer_level():
    """读出油站列表，并行处理各油站会员级别"""
    stations = list()
    rows = marketing.Station.query.all()
    for s in rows:
        update_station_customer_level.delay(s.id)
    db.session.commit()


# #################################### 定时执行任务 #################################
celery.conf.beat_schedule = {
    '处理卡券过期': {
        'task': 'app.async_task.period.parse_ticket_expire',
        'schedule': crontab(minute=10, hour=0),
    },
    '计算沉睡天数': {
        'task': 'app.async_task.period.parse_sleep_days',
        'schedule': crontab(minute=30, hour=0),
    },
    '信息定时发布': {
        'task': 'app.async_task.period.parse_info_pub',
        'schedule': crontab(minute=0, hour='*'),
    },
    '更新用户级别': {
        'task': 'app.async_task.period.update_customer_level',
        'schedule': crontab(minute=10, hour=1, day_of_month=1),
    },
}
