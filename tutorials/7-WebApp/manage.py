import sys

#linux
sys.path.append("/home/bljs/FinRL/AI4Finance/FinRL-Meta/tutorials/7-WebApp")
sys.path.append("/home/bljs/FinRL/AI4Finance/FinRL-Meta/")

#windows
# sys.path.append("D:/GoData/src/github.com/ai4finance/FinRL-Meta/tutorials/7-WebApp")
# sys.path.append("D:/GoData/src/github.com/ai4finance/FinRL-Meta")

print(sys.path)

import pickle

pickle.HIGHEST_PROTOCOL = 5
from app import create_app, db, celery
from flask_script import Manager, Shell
from flask_migrate import Migrate  # , MigrateCommand

from flask import request
from flask import make_response

# from .. import process as pro
# from .. import config_parse as config
# from .. import config_parse as cfg
import finance.process as pro
import finance.config_parse as config
import finance.config_parse as cfg

app = create_app()
manager = Manager(app)
migrate = Migrate(app, db)


def make_shell_context():
    return dict(app=app, db=db, celery=celery)


manager.add_command("shell", Shell(make_context=make_shell_context))


# manager.add_command('db', MigrateCommand)


@manager.command
def runtask():
    """Run the unit tests."""
    import os, sys
    working_dir = os.path.abspath(sys.argv[0])
    working_dir = os.path.split(working_dir)[0]
    if os.name == 'nt':
        os.system('celery -A manage.celery -P eventlet --workdir ' + working_dir + ' worker')
    else:
        os.system('celery -A manage.celery --workdir ' + working_dir + ' worker')


@manager.command
def runsche():
    """Run the unit tests."""
    import os, sys
    working_dir = os.path.abspath(sys.argv[0])
    working_dir = os.path.split(working_dir)[0]
    os.system('celery -A manage.celery --workdir ' + working_dir + ' beat')


@manager.command
def create_db():
    db.create_all()


# add ...

@app.route('/dir/list', methods=['GET'])
def dir_list():
    result = config.dir_list_get("DATA_SAVE_DIR")

    return {
        "result": result,
    }


# http://127.0.0.1:5000/dir/set?DATA_SAVE_DIR=datasets2
@app.route('/dir/set', methods=['GET', 'POST', 'PUT'])
def dir_set():
    # result = config.dir_list_get("DATA_SAVE_DIR")
    print("enter dir set")
    data = request.args.get('data')
    print(type(data))
    print(f"上传参数 {data}")
    config.dir_list_set()
    return {
        "result": 'OK',
    }


agent, mod = None, None


@app.route('/data/download', methods=['GET', 'POST', 'PUT'])
def data_download():
    time_list = cfg.time_list_get()
    # p = pro.data_process_creat(time_list['train_start_date'], time_list['trade_end_date'])  # all
    p = pro.data_process_creat(time_list['trade_start_date'], time_list['trade_end_date'])  # trade

    pro.download_data(cfg.ticker_list_get(), p)
    # pro.add_technical_factor(p)

    return {
        "result": 'OK',
    }


@app.route('/data/train', methods=['GET', 'POST', 'PUT'])
def data_train():
    time_list = cfg.time_list_get()

    print("data_train enter...")
    ticker = request.args.get('ticker')
    model = request.args.get('model')
    start = request.args.get('start')
    end = request.args.get('end')

    if start == '':
        start = time_list['train_start_date']
    if end == '':
        end = time_list['train_end_date']

    ticker_list = [ticker]
    print(ticker, model, start, end)

    p = pro.data_process_creat(start, end)  # all
    pro.download_data(ticker_list, p)

    pro.add_technical_factor(p)
    env = pro.process_env(p, start, end)

    model_name = ''
    if model == 'a2c':
        model_name = 'train_a2c_' + ticker.replace('.', '') + '_0.zip'
        agent, mod = pro.agent_a2c(env)
        trained_model = pro.data_train(agent, mod, model_name)
    elif model == 'ddpg':
        model_name = 'train_ddpg_' + ticker.replace('.', '') + '_0.zip'
        agent, mod = pro.agent_ddpg(env)
        trained_model = pro.data_train(agent, mod, model_name)

    # # agent, mod = agent_ddpg(env)
    # agent, mod = pro.agent_a2c(env)
    #
    # # ticker = cfg.ticker_list_get()[0].replace('.', '')
    # model_name_a2c = 'train_a2c_' + ticker + '_0'
    # # model_name_ddpg = 'train_ddpg_' + ticker + '_0'
    #
    # trained_model = pro.data_train(agent, mod, model_name_a2c)

    return {
        "result": 'OK',
    }


@app.route('/data/trade', methods=['GET', 'POST', 'PUT'])
def data_trade():
    org_date_start = '2001-01-01'
    time_list = cfg.time_list_get()
    # ticker_list = cfg.ticker_list_get()
    # data = request.args.get('data')
    # image = request.form["image"]
    print("data_trade enter...")
    ticker = request.args.get('ticker')
    model = request.args.get('model')
    start = request.args.get('start')
    end = request.args.get('end')

    if start == '':
        start = time_list['trade_start_date']
    if end == '':
        end = time_list['trade_end_date']

    ticker_list = [ticker]
    print(ticker, model, start, end)

    # download data
    p = pro.data_process_creat(org_date_start, end)  # trade
    pro.download_data(ticker_list, p)

    # p = pro.data_process_creat(time_list['trade_start_date'], time_list['trade_end_date'])  # all
    # p = pro.data_process_creat(start, end)  # all
    # pro.reload_data(p)
    print("----add_technical_factor", p.dataframe.shape)
    pro.add_technical_factor(p)

    env = pro.process_env(p, start, end)

    model_name = ''
    agent, mod = pro.agent_a2c(env)  # default
    if model == 'a2c':
        model_name += 'train_a2c_' + ticker.replace('.', '') + '_0.zip'
        print("model name", model_name)
        agent, mod = pro.agent_a2c(env)
        # trained_model = pro.load_model_file(mod, model_name)
        # trained_model = pro.load_model_file(mod, 'train_10k_0.zip')
    elif model == 'ddpg':
        model_name += 'train_ddpg_' + ticker.replace('.', '') + '_0.zip'
        print("model name", model_name)
        agent, mod = pro.agent_ddpg(env)
        # trained_model = pro.load_model_file(mod, model_name)

    trained_model = pro.load_model_file(mod, model_name)

    trade, account_value, actions = pro.data_predict(p, trained_model, start, end)
    # pro.back_test(trade, account_value, actions)

    # perf_stats = pro.get_returns(trade, account_value, actions)
    # annual = perf_stats['Annual return']
    # cumulative = perf_stats['Cumulative returns']
    # print("Annual and Cumulative return ", annual, cumulative)

    response = make_response(actions.to_json())
    response.headers['Access-Control-Allow-Origin'] = '*'

    return response, 200

    # return {
    #     "status": 200,
    #     "msg": actions.to_json(),
    # }


if __name__ == '__main__':
    manager.run()

# if __name__ == '__main__':
#     app = create_app()
#     app.run(host='0.0.0.0', port=88, debug=False, threaded=True)
