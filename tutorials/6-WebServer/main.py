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
    # trained_model = data_train(agent, mod, "train")

    return {
        "result": 'OK',
    }


from flask import make_response


@app.route('/data/trade', methods=['GET', 'POST', 'PUT'])
def data_trade():
    # data = request.args.get('data')
    # image = request.form["image"]
    print("data_trade enter...")
    ticker = request.args.get('ticker')
    model = request.args.get('model')
    start = request.args.get('start')
    end = request.args.get('end')

    print(ticker, model, start, end)

    time_list = cfg.time_list_get()
    p = pro.data_process_creat(time_list['trade_start_date'], time_list['trade_end_date'])  # all
    pro.reload_data(p)
    pro.add_technical_factor(p)

    env = pro.process_env(p, time_list['trade_start_date'], time_list['trade_end_date'])
    agent, mod = pro.agent_a2c(env)

    trained_model = pro.load_model_file(mod, 'train_10k_0.zip')

    trade, account_value, actions = pro.data_predict(p, trained_model, time_list['trade_start_date'],
                                                     time_list['trade_end_date'])
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
