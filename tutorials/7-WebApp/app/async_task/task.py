# -*- coding:utf-8 --*--
from datetime import datetime, timedelta
from time import sleep
from app.extensions import db,auth
from sqlalchemy import text
from celery import group, chain
from celery.schedules import crontab
from celery.exceptions import SoftTimeLimitExceeded
from app.extensions import celery
from sqlalchemy import func, text, and_
from math import floor
from datetime import datetime, timedelta, date
from decimal import Decimal
import redis, urllib
from app.config import Config
from flask import current_app
from app.config import app_settings
from lings import ORM


# url = urllib.parse.urlparse(Config.HDCODE_URL)
# hdcode_db = 0 if url.path[1:] == '' else int(url.path[1:])
# hdcode_redis = redis.Redis(host=url.hostname, port=6379 if url.port is None else url.port,
#                            db=hdcode_db, password=url.password)

url = urllib.parse.urlparse(Config.SMS_VERIFY_DB)
sms_db = 0 if url.path[1:] == '' else int(url.path[1:])
sms_redis = redis.Redis(host=url.hostname, port=6379 if url.port is None else url.port,
                         db=sms_db, password=url.password)


def execute(proc_name, data, sync=False):
    """
    异步或同步执行事务
    :param proc_name: 事务名称
    :param data: 参数，dict
    :param sync: 是否同步调用
    :return: True/False
    """
    if sync:
        return procedure[proc_name](data)
    result = procedure[proc_name].s(data).apply_async()
    res = result.get()
    return res


@celery.task(soft_time_limit=5)
def send_official_account_message(data):
    return custom_message(data)


@celery.task(soft_time_limit=5)
def code_exchange_procedure(data):
    """
    兑奖码兑奖
    :param data:
    :return:
    """
    try:
        j = data
        activity = marketing.Activity.query.filter(
            text('(uid=:uid) and (start_time <:tn) and (:tn < end_time)')).params(
            tn=datetime.now(),uid=j['uid']).first()
        if activity is None:
            return {'code': -1, 'msg': '活动已结束'}
        customer = marketing.CustomerBase.query.filter_by(openid=j['openid']).first()
        if customer is None:
            return {'code': -1, 'msg': '用户未注册'}
        cc = marketing.CustomerAccount.query.filter_by(base_id=customer.id).first()
        award_id = ''
        a = dict()
        for award in activity.settings['awards']:
            if hdcode_redis.sismember(award['id'], j['code']):
                award_id = award['id']
                a = award
                break
        if len(award_id) == 0:
            return {'code': -1, 'msg': '兑奖码无效'}
        # 兑奖记录
        ex_rec = marketing.CodeExchangeRecord()
        ex_rec.station_id = cc.station_id
        ex_rec.openid = j['openid']
        ex_rec.award_id = award_id
        ex_rec.code = j['code']
        db.session.add(ex_rec)
    #发放积分
        pir = marketing.PointIssueRecord()
        pir.amount = a['item']['point']
        cc.point_amount += pir.amount
        pir.balance = cc.point_amount
        pir.base_id = cc.base_id
        pir.base_cell = cc.cell
        pir.issue_station_id = cc.station_id
        pir.issue_station_name = cc.station_name
        pir.comment = '兑奖发放积分'
        pir.issue_source = '兑奖发放积分'
        db.session.add(cc)
        db.session.add(pir)
        #发放卡券
        for tc in a['item']['tickets']:
            for i in range(tc['num']):
                ticket_catalog = marketing.TicketCatalog.query.get(tc['ticket_catalog'])
                tr = marketing.TicketIssueRecord()
                ticket = marketing.Ticket()
                ticket.issue_station_id = cc.station_id
                ticket.issue_station_name = cc.station_name
                ticket.issue_source = '兑奖码发放'
                ticket.account = cc.account
                ticket.threshold = ticket_catalog.threshold
                ticket.ticket_catalog_id = ticket_catalog.id
                ticket.ticket_catalog_name = ticket_catalog.name
                ticket.ticket_type = ticket_catalog.type
                dn = date.today()
                ticket.start_time = dn
                ticket.end_time = ticket.start_time + timedelta(days=tc['day'])
                ticket.status = '发券'
                db.session.add(ticket)
                db.session.flush()
                tr.base_id = cc.base_id
                tr.base_cell = cc.cell
                tr.issue_source = '兑奖码发放'
                tr.issue_station_id = cc.station_id
                tr.issue_station_name = cc.station_name
                tr.ticket_id = ticket.id
                db.session.add(tr)
        db.session.commit()
        hdcode_redis.sadd(award_id + '_used', j['code'])
        hdcode_redis.srem(award_id, j['code'])
        return {'code': 0}
    except SoftTimeLimitExceeded:  # 执行超时
        db.session.rollback()
        return {'code': -1, 'msg': '执行超时'}
    except Exception as ex:
        db.session.rollback()
        return {'code': -1, 'msg': '数据库执行错误，请联系管理员'}

#查找10天指定员工服务指定客户的次数
def is_suspect(operator_id, base_cell):
    end_time = datetime.now()
    start_time = end_time + timedelta(days=-10)
    sql_params = {'operator_id':operator_id,'base_cell':base_cell,'start_time':start_time,'end_time':end_time}
    count = db.session.execute(text(app_settings.sql['count_recent_customer_by_staff']), sql_params).scalar()
    return count





@celery.task(soft_time_limit=5)
def fuel_procedure(data):
    """
    加油账务处理
    :param data:
    :return:
    """
    usr = auth.current_user
    station = marketing.Station.query.get(usr.station_id)
    j = data
    rec = marketing.FuelFillRecord()
    for i in j:
        if i != 'create_time':
            setattr(rec, i, j[i])
        else:
            setattr(rec, i, db.str2datetime(j[i]))
    if rec.fuel_type[:1] == '0':
        rec.fuel_type = '0'
    else:
        if rec.fuel_type[:3] == 'lng':
            pass
        else:
            rec.fuel_type = rec.fuel_type[:2]
    rec.station_id = usr.station_id
    rec.station_name = station.name
    d = datetime.now()
    station_code = str("{:0>2d}".format(usr.station_id))
    prefix = str('{0}{1:0>2d}{2:0>2d}'.format(d.year,d.month,d.day))
    max_sn = db.session.query(func.max(marketing.FuelFillRecord.sn)).filter(text('sn like :sn and station_id=:station_id')).params(sn=station_code + prefix + '%', station_id=usr.station_id).scalar()
    if max_sn is None:
        rec.sn = station_code + prefix + '00001'
    else:
        i = int(max_sn[-5:])+1
        rec.sn = station_code + prefix + "{:0>5d}".format(i)
    rec.operator = usr.name
    rec.operator_id = usr.id
    rec.create_time = datetime.now()
    cc = marketing.CustomerAccount.query.get(rec.account_id)
    if cc is None:
        return {'code': -1, 'msg': '用户不存在'}
    rec.base_cell = cc.cell
    rec.use_point = Decimal(j['use_point']).quantize(Decimal('0.0'))
    if (cc.point_amount is None) or (rec.use_point > cc.point_amount):
        return {'code': -1, 'msg': '账户积分余额不足'}
    else:
        cc.point_amount -= rec.use_point
    #判断是否异常
    count = is_suspect(rec.operator_id, rec.base_cell)
    if count is None:
        count = 0
    rec.suspect_count = count+1
    if (rec.suspect_count >= 2) or ((rec.fuel_type != '0') and (rec.fuel_amount >= 100)):
        rec.suspect_flag = 1
    else:
        rec.suspect_flag = 0
    #消费积分
    # pur = marketing.PointUseRecord()
    # pur.base_id = cc.base_id
    # pur.use_station_id = usr.station_id
    # pur.cell = cc.cell
    # pur.use_station_name = station.name
    # pur.amount = rec.use_point
    # pur.balance = cc.point_amount
    # pur.use_type = '加油抵扣'
    #消费加油券
    use_tickets = list()
    use_tickets_record = list()
    rec.use_ticket = 0
    rec.ticket_amount = 0
    for i in j['ticket']:
        ticket = marketing.Ticket.query.get(i)
        tr = marketing.TicketUseRecord()
        if ticket is None:
            return {'code':-1, 'msg':'卡券无效'}
        dn = datetime.now()
        if (ticket.start_time > dn) or (ticket.end_time < dn):
            return {'code': -1, 'msg': '卡券过期'}
        ticket.use_station_id = usr.station_id
        ticket.use_time = dn
        ticket.use_station_name = station.name
        ticket.status = '已使用'
        ticket.update_time = dn
        tr.base_id = cc.base_id
        tr.cell = cc.cell
        tr.use_station_id = usr.station_id
        tr.use_station_name = station.name
        tr.amount = ticket.ticket_catalog.amount
        tr.ticket_id = ticket.id
        tr.create_time = dn
        tr.use_type = '加油使用'
        tr.use_device_sn = usr.device_sn
        use_tickets.append(ticket)
        use_tickets_record.append(tr)
        rec.use_ticket += 1
        rec.ticket_amount += ticket.ticket_catalog.amount
    #发放积分、洗车券
    s = marketing.StationSettings.query.filter(text("station_id=:station_id and type='exchange_setting'")).params(station_id=usr.station_id).first()
    level_setting = marketing.StationSettings.query.filter(text("station_id=:station_id and type='level_point_setting'")).params(station_id=usr.station_id).first()
    if rec.fuel_type == '0':
        setting = s.settings['caiyou']
        level_setting = level_setting.settings['caiyou']
    else:
        if rec.fuel_type == 'lng':
            setting = s.settings['lng']
            level_setting = level_setting.settings['lng']
        else:
            setting = s.settings['qiyou']
            level_setting = level_setting.settings['qiyou']
    setting_point = setting['cash']
    pir = None
    if rec.pay_type != '充值卡支付':
        #发放积分、充值卡消费不发放积分
        times = 1
        if len(setting_point['double_rules']) > 0:
            for rule in setting_point['double_rules']:
                if (rec.pay_amount >= rule['start_value']) and (rec.pay_amount < rule['end_value']):
                    times = rule['times']
        # if (rec.pay_amount >= setting['full_value']) and (rec.isfull == 1):
        #     times = setting_point['full_times']
        #确定用户VIP级别
        level_rate = 1
        if cc.level_id > 0:
            level_rate = float(level_setting[cc.level_id-1]['times'])
        point = Decimal(round(rec.fuel_amount/setting_point['value']*setting_point['point']*times*level_rate, 1))
        rec.issue_point = point
        pir = marketing.PointIssueRecord()
        pir.base_id = cc.base_id
        pir.base_cell = cc.cell
        pir.amount = point
        pir.issue_source = '加油赠送'
        pir.issue_station_id = usr.station_id
        pir.issue_station_name = station.name
        cc.point_amount += point
        pir.balance = cc.point_amount
    if rec.fuel_type not in ["0", "lng"]: #柴油、天然气不发洗车券
        if setting['isSum'] == '累加':
            times = floor(rec.pay_amount/setting['wash_card_amount'])*setting['wash_card_value']
        else:
            if rec.pay_amount >= setting['wash_card_amount']:
                times = setting['wash_card_value']
            else:
                times = 0
        #加满多发洗车券
        if (rec.pay_amount >= setting['full_value']) and (rec.isfull == 1):
            times += setting['full_times']
    else:
        #柴油不发放洗车券
        times = 0
    tickets = list()
    trs = list()
    rec.issue_ticket = times
    for i in range(times):
        tr = marketing.TicketIssueRecord()
        ticket = marketing.Ticket()
        ticket.issue_station_id = station.id
        ticket.issue_station_name = station.name
        ticket.issue_source = '加油赠送'
        ticket.account = cc.account
        ticket.ticket_catalog_id = 1
        ticket.ticket_catalog_name = '洗车券'
        ticket.ticket_type = '洗车券'
        dn = date.today()
        ticket.start_time = dn
        ticket.end_time = ticket.start_time + timedelta(days=setting['expire_days'])
        ticket.status = '发券'
        tr.base_id = cc.base_id
        tr.base_cell = cc.cell
        tr.issue_source = '加油赠送'
        tr.issue_station_id = station.id
        tr.issue_station_name = station.name
        tickets.append(ticket)
        trs.append(tr)
    #修改原始加油记录状态
    if rec.original_record_id is not None:
        o_r = marketing.OriginalRecord.query.filter_by(id=rec.original_record_id).first()
        if o_r is not None:
            o_r.status = 1
            db.session.add(o_r)
    try:
        db.session.add(rec)
        db.session.flush()
        # pur.use_source_id = rec.id
        if pir is not None:
            pir.fuel_fill_record_id = rec.id
            db.session.add(pir)
        for i in range(times):
            db.session.add(tickets[i])
            db.session.flush()
            trs[i].ticket_id = tickets[i].id
            trs[i].fuel_fill_record_id = rec.id
            db.session.add(trs[i])
        for ut in use_tickets:
            db.session.add(ut)
        for tr in use_tickets_record:
            tr.parent_id = rec.id
            db.session.add(tr)
        db.session.add(cc)
        # db.session.add(pur)
        db.session.commit()
        #微信公众号推送消息
        cb = marketing.CustomerBase.query.filter_by(cell=cc.cell).first()
        data = dict()
        data['openid'] = cb.openid
        data['title']='你好，本次加油服务已完成'
        data['type'] = '加油'
        data['record_id'] = rec.id
        data['total'] = rec.total
        data['ticket_amount'] = rec.ticket_amount
        data['ticket_num'] = rec.use_ticket
        data['pay_amount'] = rec.pay_amount
        data['point_amount'] = 0
        execute('send_official_account_message', data, sync=True)
        return {'code': 0}
    except SoftTimeLimitExceeded:  # 执行超时
        db.session.rollback()
        return {'code': -1, 'msg': '执行超时'}
    except Exception as ex:
        db.session.rollback()
        current_app.logger.info(ex.args)
        return {'code': -1, 'msg': '数据库执行错误，请联系管理员'}


@celery.task(soft_time_limit=5)
def net_shop_exchange_procedure(data):
    """
    网上商城账务处理
    :param data:
    :return:
    """
    usr = auth.current_user
    # usr = marketing.CustomerBase.query.filter_by(cell='13888434404').first()#调试时使用
    station = marketing.Station.query.get(usr.station_id)
    j = data
    rec = marketing.NetShopRecord()
    for i in j:
        if i != 'create_time':
            setattr(rec, i, j[i])
        else:
            setattr(rec, i, db.str2datetime(j[i]))
    rec.station_id = usr.station_id
    rec.station_name = station.name
    cc = marketing.CustomerAccount.query.filter_by(cell=usr.cell).first()
    rec.account_id = cc.id
    d = datetime.now()
    station_code = str("{:0>2d}".format(usr.station_id))
    prefix = str('{0}{1:0>2d}{2:0>2d}'.format(d.year,d.month,d.day))
    max_sn = db.session.query(func.max(marketing.FuelFillRecord.sn)).filter(text('sn like :sn and station_id=:station_id')).params(sn=station_code + prefix + '%', station_id=usr.station_id).scalar()
    if max_sn is None:
        rec.sn = station_code + prefix + '00001'
    else:
        i = int(max_sn[-5:])+1
        rec.sn = station_code + prefix + "{:0>5d}".format(i)
    # rec.operator = usr.name
    rec.create_time = datetime.now()
    rec.base_cell = cc.cell
    rec.use_point = Decimal(j['use_point']).quantize(Decimal('0.0'))
    if (cc.point_amount is None) or (rec.use_point > cc.point_amount):
        return {'code': -1, 'msg': '账户积分余额不足'}
    else:
        cc.point_amount -= rec.use_point
    pur = marketing.PointUseRecord()
    pur.base_id = cc.base_id
    pur.use_station_id = usr.station_id
    pur.cell = cc.cell
    pur.use_station_name = station.name
    pur.amount = rec.use_point
    pur.balance = cc.point_amount
    pur.use_type = '网上商城换购'
    #发券
    tickets = list()
    trs = list()
    for i in range(j['num']):
        item = marketing.StoreItem.query.filter_by(id=j['item_id']).first()
        for ticket_catalog in item.detail:
            for n in range(ticket_catalog['num']):
                tr = marketing.TicketIssueRecord()
                ticket = marketing.Ticket()
                ticket.issue_station_id = station.id
                ticket.issue_station_name = station.name
                ticket.issue_source = '网上商城换购'
                ticket.account = cc.account
                ticket.threshold = ticket_catalog['ticket_catalog']['threshold']
                ticket.ticket_catalog_id = ticket_catalog['ticket_catalog']['id']
                ticket.ticket_catalog_name = ticket_catalog['ticket_catalog']['name']
                ticket.ticket_type = ticket_catalog['ticket_catalog']['type']
                dn = date.today()
                ticket.start_time = dn
                ticket.end_time = ticket.start_time + timedelta(days=ticket_catalog['day'])
                ticket.status = '发券'
                tr.base_id = cc.base_id
                tr.base_cell = cc.cell
                tr.issue_source = '网上商城换购'
                tr.issue_station_id = station.id
                tr.issue_station_name = station.name
                tickets.append(ticket)
                trs.append(tr)
    try:
        db.session.add(rec)
        db.session.flush()
        pur.use_source_id = rec.id
        for i in range(len(tickets)):
            db.session.add(tickets[i])
            db.session.flush()
            trs[i].ticket_id = tickets[i].id
            trs[i].fuel_fill_record_id = rec.id
            db.session.add(trs[i])
        db.session.add(cc)
        db.session.add(pur)
        db.session.commit()
        return {'code': 0}
    except SoftTimeLimitExceeded:  # 执行超时
        db.session.rollback()
        return {'code': -1, 'msg': '执行超时'}
    except Exception as ex:
        db.session.rollback()
        return {'code': -1, 'msg': '数据库执行错误，请联系管理员'}


@celery.task(soft_time_limit=5)
def shop_procedure(data):
    """
    便利店账务处理
    :param data:
    :return:
    """
    usr = auth.current_user
    station = marketing.Station.query.get(usr.station_id)
    j = data
    rec = marketing.ShopRecord()
    for i in j:
        if i != 'create_time':
            setattr(rec, i, j[i])
        else:
            setattr(rec, i, db.str2datetime(j[i]))
    rec.station_id = usr.station_id
    rec.station_name = station.name
    rec.device_sn = usr.device_sn
    rec.operator = usr.name
    rec.operator_id = usr.id
    d = datetime.now()
    station_code = str("{:0>2d}".format(usr.station_id))
    prefix = str('{0}{1:0>2d}{2:0>2d}'.format(d.year,d.month,d.day))
    max_sn = db.session.query(func.max(marketing.ShopRecord.sn)).filter(text('sn like :sn and station_id=:station_id')).params(sn=station_code + prefix + '%', station_id=usr.station_id).scalar()
    if max_sn is None:
        rec.sn = station_code + prefix + '00001'
    else:
        i = int(max_sn[-5:])+1
        rec.sn = station_code + prefix + "{:0>5d}".format(i)
    rec.create_time = datetime.now()
    cc = marketing.CustomerAccount.query.get(rec.account_id)
    rec.base_cell = cc.cell
    rec.use_point = Decimal(j['use_point']).quantize(Decimal('0.0'))
    if (cc.point_amount is None) or (rec.use_point > cc.point_amount):
        return {'code': -1, 'msg': '账户积分余额不足{0},{1}'.format(rec.use_point,cc.point_amount)}
    else:
        cc.point_amount -= rec.use_point
    #记录明细
    for i in j['items']:
        rd = marketing.ShopRecordDetail()
        rd.station_id = rec.station_id
        rd.operator = rec.operator
        rd.operator_id = rec.operator_id
        rd.price = i['price']
        rd.count = i['num']
        rd.total = rd.price*rd.count
        rd.station_store_item_id = i['id']
        rd.station_store_item_name = i['name']
        rd.base_cell = rec.base_cell
        db.session.add(rd)
    #消费卡券
    use_tickets = list()
    use_tickets_record = list()
    rec.use_ticket = 0
    rec.ticket_amount = 0
    for i in j['ticket']:
        ticket = marketing.Ticket.query.get(i)
        tr = marketing.TicketUseRecord()
        if ticket is None:
            return {'code':-1, 'msg':'卡券无效'}
        if ticket.status == '已使用':
            return {'code': -1, 'msg': '卡券无效'}
        dn = datetime.now()
        if (ticket.start_time > dn) or (ticket.end_time < dn):
            return {'code': -1, 'msg': '卡券过期'}
        ticket.use_station_id = usr.station_id
        ticket.use_time = dn
        ticket.use_station_name = station.name
        ticket.status = '已使用'
        ticket.update_time = dn
        tr.base_id = cc.base_id
        tr.cell = cc.cell
        tr.use_station_id = usr.station_id
        tr.use_station_name = station.name
        tr.use_device_sn = usr.device_sn
        tr.use_type = '便利店消费'
        tr.amount = ticket.ticket_catalog.amount
        tr.ticket_id = ticket.id
        tr.create_time = dn
        use_tickets.append(ticket)
        use_tickets_record.append(tr)
        rec.use_ticket += 1
        rec.ticket_amount += ticket.ticket_catalog.amount
    try:
        db.session.add(rec)
        db.session.flush()
        if rec.use_point > 0:
            pur = marketing.PointUseRecord()
            pur.base_id = cc.base_id
            pur.use_station_id = usr.station_id
            pur.cell = cc.cell
            pur.use_station_name = station.name
            pur.amount = rec.use_point
            pur.balance = cc.point_amount
            pur.use_device_sn = usr.device_sn
            pur.use_type = '便利店消费'
            pur.use_source_id = rec.id
            db.session.add(pur)
        for ut in use_tickets:
            db.session.add(ut)
        for tr in use_tickets_record:
            tr.parent_id = rec.id
            db.session.add(tr)
        db.session.add(cc)
        db.session.commit()
        #微信公众号推送消息
        cb = marketing.CustomerBase.query.filter_by(cell=cc.cell).first()
        data = dict()
        data['openid'] = cb.openid
        data['title'] = '你好，便利店消费清单如下'
        data['type'] = '便利店'
        data['record_id'] = rec.id
        data['total'] = rec.total
        data['ticket_amount'] = rec.ticket_amount
        data['ticket_num'] = rec.use_ticket
        data['pay_amount'] = rec.pay_amount
        data['point_amount'] = rec.use_point
        execute('send_official_account_message', data, sync=True)
        return {'code': 0}
    except SoftTimeLimitExceeded:  # 执行超时
        db.session.rollback()
        return {'code': -1, 'msg': '执行超时'}
    except Exception as ex:
        db.session.rollback()
        return {'code': -1, 'msg': '数据库执行错误，请联系管理员'}


@celery.task(soft_time_limit=5)
def issue_point_batch(data):
    """
    发放积分
    :param data:
    :return:
    """
    j = data['json']
    cc = data['customer']
    try:
        pir = marketing.PointIssueRecord()
        pir.amount = j['point_amount']
        cc.point_amount += pir.amount
        pir.balance = cc.point_amount
        pir.base_id = cc.base_id
        pir.base_cell = cc.cell
        pir.issue_station_id = cc.station_id
        pir.issue_station_name = cc.station_name
        pir.comment = j['comment']
        pir.issue_source = '手动批量发放积分'
        db.session.add(cc)
        db.session.add(pir)
        db.session.commit()
        return {'code': 0}
    except SoftTimeLimitExceeded:       # 执行超时
        db.session.rollback()
        return {'code': -1, 'msg': '执行超时'}
    except Exception as ex:
        db.session.rollback()
        current_app.logger.info(ex.args)
        return {'code': -1, 'msg': '数据库执行错误，请联系管理员'}


@celery.task(soft_time_limit=5)
def issue_ticket_batch(data):
    """
    发放卡券
    :param data:
    :return:
    """
    j = data['json']
    cc = data['customer']
    try:
        #发放洗车券
        if 'carwash' in j:
            for i in range(j['carwash']['num']):
                tr = marketing.TicketIssueRecord()
                ticket = marketing.Ticket()
                ticket.issue_station_id = cc.station_id
                ticket.issue_station_name = cc.station_name
                ticket.issue_source = '手动批量发放'
                ticket.account = cc.account
                ticket.ticket_catalog_id = 1
                ticket.ticket_catalog_name = '洗车券'
                ticket.ticket_type = '洗车券'
                dn = date.today()
                ticket.start_time = dn
                ticket.end_time = ticket.start_time + timedelta(days=j['carwash']['day'])
                ticket.status = '发券'
                db.session.add(ticket)
                db.session.flush()
                tr.base_id = cc.base_id
                tr.base_cell = cc.cell
                tr.issue_source = '手动批量发放'
                tr.issue_station_id = cc.station_id
                tr.issue_station_name = cc.station_name
                tr.ticket_id = ticket.id
                db.session.add(tr)
        #发放其他卡券
        for tc in j['ticket']:
            for i in range(int(tc['num'])):
                ticket_catalog = marketing.TicketCatalog.query.get(tc['ticket_catalog'])
                tr = marketing.TicketIssueRecord()
                ticket = marketing.Ticket()
                ticket.issue_station_id = cc.station_id
                ticket.issue_station_name = cc.station_name
                ticket.issue_source = '手动批量发放'
                ticket.account = cc.account
                ticket.threshold = ticket_catalog.threshold
                ticket.ticket_catalog_id = ticket_catalog.id
                ticket.ticket_catalog_name = ticket_catalog.name
                ticket.ticket_type = ticket_catalog.type
                dn = date.today()
                ticket.start_time = dn
                ticket.end_time = ticket.start_time + timedelta(days=tc['day'])
                ticket.status = '发券'
                db.session.add(ticket)
                db.session.flush()
                tr.base_id = cc.base_id
                tr.base_cell = cc.cell
                tr.issue_source = '手动批量发放'
                tr.issue_station_id = cc.station_id
                tr.issue_station_name = cc.station_name
                tr.ticket_id = ticket.id
                db.session.add(tr)
        db.session.commit()
        return {'code': 0}
    except SoftTimeLimitExceeded:       # 执行超时
        db.session.rollback()
        return {'code': -1, 'msg': '执行超时'}
    except Exception as ex:
        db.session.rollback()
        current_app.logger.info(ex.args)
        return {'code': -1, 'msg': '数据库执行错误，请联系管理员'}


def before_account_insert(customer_account):
    d = datetime.now()
    station_code = str("{:0>2d}".format(customer_account.station_id))
    prefix = str('{0}{1:0>2d}{2:0>2d}'.format(d.year,d.month,d.day))
    max_sn = db.session.query(func.max(marketing.CustomerAccount.account)).filter(text('account like :account and station_id=:station_id')).params(account=station_code + prefix + '%', station_id=customer_account.station_id).scalar()
    if max_sn is None:
        customer_account.account = station_code + prefix + '001'
    else:
        i = int(max_sn[-3:])+1
        customer_account.account = station_code + prefix + "{:0>3d}".format(i)


@celery.task(soft_time_limit=5)
def register_procedure(data):
    j = data
    if j['code'] != 'ynl565':
        c = sms_redis.get(j['cell'])
        if c is None:
            return {'code': -1, 'msg':'验证码失效'}
        code = bytes.decode(c)
        if j['code'] != code:
            return {'code': -1, 'msg': '验证码错误'}
    cb = marketing.CustomerBase.query.filter_by(cell=j['cell']).first()
    if 'invite_code' in j:
        staff = marketing.Staff.query.filter_by(invite_code=j['invite_code']).first()
    else:
        staff = None
    # if staff is None:
    #     return {'code': -1, 'msg': '邀请码错误'}
    if cb is not None:
        return {'code': -1, 'msg': '该手机用户已在系统内注册'}
    try:
        s = marketing.Station.query.get(j['station_id'])
        cb = marketing.CustomerBase()
        cb.openid = j['openid']
        cb.cell = j['cell']
        cb.plate = j['plate'][0] + j['plate'][1:].upper()
        cb.level = 1
        cb.station_id = j['station_id']
        cb.station_name = s.name
        cb.update_time = datetime.now()
        db.session.add(cb)
        db.session.flush()
        cc = marketing.CustomerAccount()
        cc.station_id = cb.station_id
        cc.station_name = cb.station_name
        cc.base_id = cb.id
        cc.account_balance = 0
        cc.point_amount = 0
        cc.cell = cb.cell
        cc.level = cb.level
        cc.sleep_days = 0
        before_account_insert(cc)
        cc.update_time = datetime.now()
        db.session.add(cc)
        db.session.flush()
        # 新用户发放积分、卡券
        setting = marketing.StationSettings.query.filter(text("station_id=:station_id and type='newuser_setting'")) \
            .params(station_id=cb.station_id).first()
        if setting is not None:
            #发放积分
            p = Decimal(setting.settings['point_amount'])
            cc.point_amount += p
            pir = marketing.PointIssueRecord()
            pir.base_id = cc.base_id
            pir.base_cell = cc.cell
            pir.amount = p
            pir.issue_source = '新用户赠送'
            pir.issue_station_id = cb.station_id
            pir.issue_station_name = cb.station_name
            pir.balance = cc.point_amount
            db.session.add(pir)
            #发放洗车券
            # tickets = list()
            # trs = list()
            for i in range(int(setting.settings['car_amount'])):
                tr = marketing.TicketIssueRecord()
                ticket = marketing.Ticket()
                ticket.issue_station_id = cc.station_id
                ticket.issue_station_name = cc.station_name
                ticket.issue_source = '新用户注册赠送'
                ticket.account = cc.account
                ticket.ticket_catalog_id = 1
                ticket.ticket_catalog_name = '洗车券'
                ticket.ticket_type = '洗车券'
                dn = date.today()
                ticket.start_time = dn
                ticket.end_time = ticket.start_time + timedelta(days=int(setting.settings['car_indate']))
                ticket.status = '发券'
                db.session.add(ticket)
                db.session.flush()
                tr.base_id = cc.base_id
                tr.base_cell = cc.cell
                tr.issue_source = '新用户注册赠送'
                tr.issue_station_id = cc.station_id
                tr.issue_station_name = cc.station_name
                tr.ticket_id = ticket.id
                db.session.add(tr)
                # tickets.append(ticket)
                # trs.append(tr)
            #发放其它卡券
            if isinstance(setting.settings['ticket'],list):
                for ticket_catalog in setting.settings['ticket']:
                    dn = datetime.now()
                    if (db.str2datetime(ticket_catalog['issue_time'][0]) < dn) and (db.str2datetime(ticket_catalog['issue_time'][1]) > dn):
                        for i in range(int(ticket_catalog['amount'])):
                            tr = marketing.TicketIssueRecord()
                            ticket = marketing.Ticket()
                            ticket.issue_station_id = cc.station_id
                            ticket.issue_station_name = cc.station_name
                            ticket.issue_source = '新用户注册赠送'
                            ticket.account = cc.account
                            ticket.ticket_catalog_id = ticket_catalog['ticket_catalog']['id']
                            ticket.ticket_catalog_name = ticket_catalog['ticket_catalog']['name']
                            ticket.ticket_type = ticket_catalog['ticket_catalog']['type']
                            ticket.start_time = dn
                            ticket.end_time = ticket.start_time + timedelta(days=int(ticket_catalog['indate_val']))
                            ticket.status = '发券'
                            db.session.add(ticket)
                            db.session.flush()
                            tr.base_id = cc.base_id
                            tr.base_cell = cc.cell
                            tr.issue_source = '新用户注册赠送'
                            tr.issue_station_id = cc.station_id
                            tr.issue_station_name = cc.station_name
                            tr.ticket_id = ticket.id
                            db.session.add(tr)
        rr = marketing.RegisterRecord()
        rr.base_id = cb.id
        if staff is not None:
            rr.invite_code = staff.invite_code
            rr.staff_name = staff.name
        db.session.add(rr)
        db.session.commit()
        if 'invite_id' in j:
            invite_procedure(j['invite_id'])
        return {'code': 0}
    except SoftTimeLimitExceeded:       # 执行超时
        db.session.rollback()
        return {'code': -1, 'msg': '执行超时'}
    except Exception as ex:
        db.session.rollback()
        if hasattr(ex,'message'):
            msg = ex.message
        else:
            msg = ex.args[0]
        return {'code': -1, 'msg': msg}


# 邀请有礼
@celery.task(soft_time_limit=5)
def invite_procedure(invite_id):
        # 获取邀请者信息
    try:
        invite_user = marketing.CustomerBase.query.get(invite_id)
        if invite_user is None:
            return {'code': -1, 'msg': '该用户未找到'}
        cc = marketing.CustomerAccount.query.filter_by(base_id=invite_id).first()
        # 新用户发放积分、卡券
        setting = marketing.StationSettings.query.filter(text("station_id=:station_id and type='inviteuser_setting'")) \
            .params(station_id=invite_user.station_id).first()
        if setting is not None:
            #发放积分
            p = Decimal(setting.settings['point_amount'])
            cc.point_amount += p
            pir = marketing.PointIssueRecord()
            pir.base_id = cc.base_id
            pir.base_cell = cc.cell
            pir.amount = p
            pir.issue_source = '邀请有礼赠送'
            pir.issue_station_id = invite_user.station_id
            pir.issue_station_name = invite_user.station_name
            pir.balance = cc.point_amount
            db.session.add(pir)
            #发放洗车券
            # tickets = list()
            # trs = list()
            for i in range(int(setting.settings['car_amount'])):
                tr = marketing.TicketIssueRecord()
                ticket = marketing.Ticket()
                ticket.issue_station_id = cc.station_id
                ticket.issue_station_name = cc.station_name
                ticket.issue_source = '邀请用户赠送'
                ticket.account = cc.account
                ticket.ticket_catalog_id = 1
                ticket.ticket_catalog_name = '洗车券'
                ticket.ticket_type = '洗车券'
                dn = date.today()
                ticket.start_time = dn
                ticket.end_time = ticket.start_time + timedelta(days=int(setting.settings['car_indate']))
                ticket.status = '发券'
                db.session.add(ticket)
                db.session.flush()
                tr.base_id = cc.base_id
                tr.base_cell = cc.cell
                tr.issue_source = '邀请用户赠送'
                tr.issue_station_id = cc.station_id
                tr.issue_station_name = cc.station_name
                tr.ticket_id = ticket.id
                db.session.add(tr)
                # tickets.append(ticket)
                # trs.append(tr)
            #发放其它卡券
            if isinstance(setting.settings['ticket'],list):
                for ticket_catalog in setting.settings['ticket']:
                    dn = datetime.now()
                    if (db.str2datetime(ticket_catalog['issue_time'][0]) < dn) and (db.str2datetime(ticket_catalog['issue_time'][1]) > dn):
                        for i in range(int(ticket_catalog['amount'])):
                            tr = marketing.TicketIssueRecord()
                            ticket = marketing.Ticket()
                            ticket.issue_station_id = cc.station_id
                            ticket.issue_station_name = cc.station_name
                            ticket.issue_source = '邀请用户赠送'
                            ticket.account = cc.account
                            ticket.ticket_catalog_id = ticket_catalog['ticket_catalog']['id']
                            ticket.ticket_catalog_name = ticket_catalog['ticket_catalog']['name']
                            ticket.ticket_type = ticket_catalog['ticket_catalog']['type']
                            ticket.start_time = dn
                            ticket.end_time = ticket.start_time + timedelta(days=int(ticket_catalog['indate_val']))
                            ticket.status = '发券'
                            db.session.add(ticket)
                            db.session.flush()
                            tr.base_id = cc.base_id
                            tr.base_cell = cc.cell
                            tr.issue_source = '邀请用户赠送'
                            tr.issue_station_id = cc.station_id
                            tr.issue_station_name = cc.station_name
                            tr.ticket_id = ticket.id
                            db.session.add(tr)
            db.session.commit()
    except SoftTimeLimitExceeded:       # 执行超时
        db.session.rollback()
        return {'code': -1, 'msg': '执行超时'}
    except Exception as ex:
        db.session.rollback()
        if hasattr(ex,'message'):
            msg = ex.message
        else:
            msg = ex.args[0]
        return {'code': -1, 'msg': msg}

@celery.task
def send_point_message(data):
    for customer in data['customers']:
        customer_base = marketing.CustomerBase.query.get(customer)
        customer_account = marketing.CustomerAccount.query.filter_by(base_id=customer).first()
        obj = {
            "openid": customer_base.openid,
            "title": "你的积分发生变更，请到会员中心查看",
            "issue_point": data['point_amount'],
            "point": customer_account.point_amount,
            "type": '手动赠送积分'
        }
        template_message(CustomMessage.point_send(obj))
    return True

@celery.task
def send_ticket_message(data):
    for customer in data['customers']:
        customer_base = marketing.CustomerBase.query.get(customer)
        customer_account = marketing.CustomerAccount.query.filter_by(base_id=customer).first()
        obj = {
            "openid": customer_base.openid,
            "title": "手动赠送积分",
            "issue_point": data['point_amount'],
            "point": customer_account.point_amount,
            "type": '赠送积分'
        }
        template_message(CustomMessage.point_send(obj))
    return True

procedure = {
    'fuel': fuel_procedure,
    'register': register_procedure,
    'issue_point_batch':issue_point_batch,
    'issue_ticket_batch': issue_ticket_batch,
    'code_exchange_procedure':code_exchange_procedure,
    'shop_procedure':shop_procedure,
    'net_shop_exchange':net_shop_exchange_procedure,
    'send_official_account_message':send_official_account_message,
    'invite_procedure': invite_procedure,
    'send_point_message':send_point_message
}

# #################################### 定时执行任务 #################################
# celery.conf.beat_schedule = {
#     '评估所有监测点并更新矿库状态': {
#         'task': 'app.async_task.task.eval_all',
#         'schedule': crontab(minute=0, hour='*/2'),
#     },
#     '每5分钟检查自动发送短信': {
#         'task': 'app.async_task.task._auto_send_sms',
#         'schedule': crontab(minute='*/5', hour='*'),
#     },
# }
