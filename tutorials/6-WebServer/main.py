from flask import Flask
from flask import request
import config_parse as config
import process as pro
import config_parse as cfg

# https://blog.csdn.net/qq_26086231/article/details/124787335
# 需要导入模块: from flask import request [as 别名]
# 或者: from flask.request import form [as 别名]
app = Flask(__name__)


@app.route('/testPost/', methods=['POST'])
def test_post():
    # image = request.form["image"]
    #
    return {
        "result": "OK"
    }


@app.route('/test/', methods=['GET'])
def test_request():
    return {
        "result": "OK",
    }


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
    p = pro.data_process_creat(time_list['train_start_date'], time_list['train_end_date'])  # all

    pro.add_technical_factor(p)
    env = pro.process_env(p, time_list['train_start_date'], time_list['train_end_date'])
    # agent, mod = agent_ddpg(env)
    agent, mod = pro.agent_a2c(env)

    ticker = cfg.ticker_list_get()[0].replace('.', '')
    model_name_a2c = 'train_a2c_' + ticker + '_0'
    # model_name_ddpg = 'train_ddpg_' + ticker + '_0'

    trained_model = pro.data_train(agent, mod, model_name_a2c)

    return {
        "result": 'OK',
    }


from flask import make_response


@app.route('/data/trade', methods=['GET', 'POST', 'PUT'])
def data_trade():
    time_list = cfg.time_list_get()
    # ticker_list = cfg.ticker_list_get()
    # data = request.args.get('data')
    # image = request.form["image"]
    print("data_trade enter...")
    ticker = request.args.get('ticker')
    model = request.args.get('model')
    start = request.args.get('start')
    end = request.args.get('end')

    if start=='':
        start = time_list['trade_start_date']
    if end == '':
        end = time_list['trade_end_date']

    ticker_list = [ticker]
    print(ticker, model, start, end)

    #download data
    p = pro.data_process_creat(start, end)  # trade
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
    response = make_response(actions.to_json())
    response.headers['Access-Control-Allow-Origin'] = '*'

    return response, 200

    # return {
    #     "status": 200,
    #     "msg": actions.to_json(),
    # }


if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=5000)
