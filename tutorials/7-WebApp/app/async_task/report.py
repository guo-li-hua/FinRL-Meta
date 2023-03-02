from datetime import datetime, timedelta
from celery.exceptions import SoftTimeLimitExceeded
from app.extensions import celery
from app.extensions import api, auth, db
from app.public.func import str2datetime
from flask import current_app, request
from app.config import app_settings
from sqlalchemy import text
import pandas as pd
import numpy as np


@celery.task(soft_time_limit=30)
def station_3KU(data):
    try:
        if data is None or 'start_time' not in data:
            start_time = datetime.now()-timedelta(days=365)
            start_time = datetime(start_time.year, start_time.month, 1)
            end_time = datetime.now()
            date_format = '%Y-%m'
        else:
            start_time = str2datetime(data['start_time'])
            end_time = str2datetime(data['end_time']) + timedelta(days=1)
            date_format = '%Y-%m-%d' if data['date_format'] == 'D' else '%Y-%m-%d %H:00'

        station_id = auth.current_user.station_id
        # station_id = 1
        new_customer = pd.read_sql(text(app_settings.sql['new_customer_count']), db.engine, index_col='年月',
                                   params=dict(station_id=station_id, start_time=start_time, end_time=end_time, date_format=date_format))
        rpt = pd.read_sql(text(app_settings.sql['3KU_station']), db.engine, index_col='年月',
                          params=dict(station_id=station_id, start_time=start_time, end_time=end_time, date_format=date_format))
        rpt = rpt.join(new_customer)
        rpt['客频'] = rpt['单数']/rpt['来客数']
        rpt['客单价'] = rpt['金额'] / rpt['单数']
        rpt['柴油环比'] = rpt['柴油'].diff()
        rpt['汽油环比'] = rpt['汽油'].diff()
        rpt['柴油环比'] = rpt['柴油环比']/(rpt['柴油']-rpt['柴油环比'])*100
        rpt['汽油环比'] = rpt['汽油环比']/(rpt['汽油']-rpt['汽油环比'])*100
        rpt = rpt.round(2).sort_index(ascending=False).replace([np.inf, -np.inf], np.nan).fillna(0)
        return rpt.reset_index()
    except SoftTimeLimitExceeded:
        return None
    except Exception as ex:
        print(ex)
        return None


@celery.task(soft_time_limit=30)
def overview_sale_sum(data):
    try:
        start_time = str2datetime(data['start_time'])
        end_time = str2datetime(data['end_time']) + timedelta(days=1)

        issue = pd.read_sql(text(app_settings.sql['overview_issue_point']), db.engine, index_col='加油站',
                            params=dict(start_time=start_time, end_time=end_time))
        use = pd.read_sql(text(app_settings.sql['overview_use_point']), db.engine, index_col='加油站',
                          params=dict(start_time=start_time, end_time=end_time))
        df = issue.join(use)
        df1 = pd.DataFrame(columns=[['积分'] * len(df.columns), df.columns])
        df1[df1.columns] = df[df.columns]

        df = pd.read_sql(text(app_settings.sql['overview_shop_sum']), db.engine, index_col='加油站',
                         params=dict(start_time=start_time, end_time=end_time))
        df2 = pd.DataFrame(columns=[['便利店']*len(df.columns), df.columns])
        df2[df2.columns] = df[df.columns]

        df = pd.read_sql(text(app_settings.sql['overview_fuel_sum']), db.engine, index_col='加油站',
                         params=dict(start_time=start_time, end_time=end_time))
        df3 = pd.DataFrame(columns=[['油料销量（L）']*len(df.columns), df.columns])
        df3[df3.columns] = df[df.columns]

        df = df1.join(df2).join(df3).fillna(0)
        if not df.empty:
            df.loc['合计'] = df.sum()
        return df
    except SoftTimeLimitExceeded:
        return None
    except Exception as ex:
        print(ex)
        return None