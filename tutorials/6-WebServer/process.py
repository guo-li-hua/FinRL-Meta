import os
import sys

sys.path.append("../../")
sys.path.append("../../../FinRL-Meta")

import json
import matplotlib
import matplotlib.pyplot as plt
import datetime
import warnings
import pandas as pd
import config_parse as cfg
import common
from meta.data_processor import DataProcessor
# from common import check_and_make_directories
from meta.data_processors.tushare import Tushare, ReturnPlotter
from meta.env_stock_trading.env_stocktrading_China_A_shares import (
    StockTradingEnv,
)

from agents.stablebaselines3_models import DRLAgent
from stable_baselines3.common.logger import configure as logconfigure

from meta import config
import pyfolio
from pyfolio import timeseries
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from IPython import display

# 系统参数配置
print(matplotlib.get_backend())
# matplotlib.use('TkAgg')  # TkAgg  'module://backend_interagg'

warnings.filterwarnings("ignore")
pd.options.display.max_columns = None

# 目录结构创建
dirs = cfg.dir_list_get()
print("type", dirs),

common.check_and_make_directories(
    [dirs['data_save_dir'], dirs['trained_model_dir'], dirs['tensorboard_log_dir'], dirs['results_dir']]
)

# 加载数据、时间配置项
# ticker_list = cfg.ticker_list_get()
# time_list = cfg.time_list_get()
# print(time_list)

# cfg.ticker_list_add("123")
# cfg.time_list_set('TRAIN_START_DATE','2014-02-02')
# 1个月  3个月

kwargs = {}
kwargs["token"] = "3082a6ea9b417fdfb3d32e4af9cde1e7af610246378c131ef24ab062"

#SSE 50：000016
#沪深300  399300
# reference_tickers = ['000016.SZ', '399300.SZ']
# reference_tickers = ['300999.SZ', '300121.SZ']
reference_tickers = []

# "27080ec403c0218f96f388bca1b1d85329d563c91a43672239619ef5"

def data_process_creat(start, end):
    time_list = cfg.time_list_get()
    data_source = cfg.curent_data_source_get()
    p = DataProcessor(
        data_source=data_source,  # "tushare",
        start_date=start,
        end_date=end,
        time_interval=time_list['time_interval'],
        **kwargs,
    )
    return p


# 下载数据
def download_data(tickers, p):
    file_path, file_dir, file_name = common.cache_file(tickers, p.start_date, p.end_date)
    p.run_download(tickers, cache=True, file_dir=file_dir, file_name=file_name)

    # price_array, tech_array, turbulence_array
    p.clean_data()

    return p


# 加载数据
def reload_data(p):
    file_path, file_dir, file_name = common.cache_file(p.start_date, p.end_date)
    p.run_fileload(file_dir=file_dir, file_name=file_name)

    p.clean_data()
    return p


def add_technical_factors(p):
    indicator = cfg.indicators_get()
    factor_cfg = cfg.factors_cfg_get()

    # p.clean_data()
    # p.add_turbulence()
    # print("before add_technical_indicator", p.dataframe.shape)
    p.add_technical_indicator(indicator)
    # print("after add_technical_indicator", p.dataframe.shape)

    p.add_technical_factor(factor_cfg)
    # print("after add_technical_factor", p.dataframe.shape)

    # 这里可以下载指标数据/或者指标数据转换
    p.add_reference_tickers(reference_tickers)
    print(f"p.dataframe: {p.dataframe}")
    p.clean_data()


### Train
def process_env(p, start, end):
    time_list = cfg.time_list_get()
    indicators = cfg.indicators_get()
    factors = cfg.factors_get()

    # train = common.data_process(p, time_list['train_start_date'], time_list['train_end_date'], 0)
    train = common.data_process(p, start, end, 20)
    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension * (len(indicators) + len(factors) + 2)  + 1
    # print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    print(train)
    env_kwargs = common.env_kwargs(train)
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    print(f"print(type(env_train)): {print(type(env_train))}")
    return env_train


def agent_ddpg(env_train):
    ## DDPG
    agent = DRLAgent(env=env_train)
    DDPG_PARAMS = {
        "batch_size": 2560,
        "buffer_size": 50000,
        "learning_rate": 0.0005,
        "action_noise": "normal",
    }
    # pi=[<actor network architecture>]
    # qf=[<critic network architecture>]
    POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))
    model_ddpg = agent.get_model(
        "ddpg", model_kwargs=DDPG_PARAMS, policy_kwargs=POLICY_KWARGS
    )
    return agent, model_ddpg


def agent_a2c(env_train):
    ## A2C
    agent = DRLAgent(env=env_train)
    model_a2c = agent.get_model("a2c")
    return agent, model_a2c


def agent_td3(env_train):
    ## TD3
    agent = DRLAgent(env=env_train)
    model_td3 = agent.get_model("td3")
    return agent, model_td3


def agent_sac(env_train):
    ## SAC
    agent = DRLAgent(env=env_train)
    model_sac = agent.get_model("sac")
    return agent, model_sac


def agent_ppo(env_train):
    ## PPO
    agent = DRLAgent(env=env_train)
    model_ppo = agent.get_model("ppo")
    return agent, model_ppo


def data_train(agent, model, name):
    trained_model = agent.train_model(
        model=model, tb_log_name=name, total_timesteps=5000
    )
    return trained_model

    # trained_ddpg = agent.train_model(
    #     model=model_ddpg, tb_log_name="ddpg", total_timesteps=10000
    # )
    # trained_a2c = agent.train_model(
    #     model=model_a2c, tb_log_name="a2c", total_timesteps=50000
    # )


def load_model_file(model, name):
    dir_list = cfg.dir_list_get()

    model_file = f"{dir_list['trained_model_dir']}/{name}"
    print("model_file:", model_file)

    if os.path.isfile(model_file):
        trained_model = model.load(model_file)
        return trained_model

    # model_file_name = "a2c_50k_0.zip"
    # trained_a2c_read = model_a2c.load(f"{config.TRAINED_MODEL_DIR}/{model_file_name}")
    return None


def data_predict(p, model, start, end):
    ### Trade
    time_list = cfg.time_list_get()
    # print(p.dataframe)
    trade = p.data_split(p.dataframe, start, end)
    print("data_predict.......")
    # print(trade)
    env_kwargs = common.env_kwargs(trade)
    e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)

    # df_account_value, df_actions = DRLAgent.DRL_prediction(
    #     model=trained_ddpg, environment=e_trade_gym
    # )

    # model.eval()
    # DRLAgent.predict_generator

    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=model, environment=e_trade_gym
    )
    print(df_account_value)
    print(df_actions)

    #
    # action_start = df_actions['date'].values[0]
    # action_end = df_actions['date'].values[-1]
    # print(action_start, action_end)
    #
    # # tur_list = p.dataframe.loc[action_start:action_end, 'turbulence']
    # # tur_list = p.dataframe.loc['2022-11-02':'2023-01-10', 'turbulence']
    #
    # tur_data = p.data_split(p.dataframe, action_start, action_end)
    # # print("tur_data:", tur_data)
    # tur_list = tur_data['turbulence']
    # tmp_turbulence_threshold = 0.2

    # print(tur_list)

    # if tmp_turbulence_threshold is not None:
    #     for i in range(len(df_actions)):
    #         print(tur_list[i], tmp_turbulence_threshold)
    #         if tur_list[i] < tmp_turbulence_threshold:
    #             print("turbulence < threshold-", i)
    #             df_actions.iloc[i, 1] = 0
    #
    #     print("df_actions", df_actions)

    return trade, df_account_value, df_actions


def plt_loss():
    # 读取数据
    file_name = "results/log.progress.csv"
    data = pd.read_csv("results/log/progress.csv")  # , parse_dates=['Date']

    fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(data['time/iterations'], data['train/reward'], label='reward')

    # 定义画布并添加子图
    # fig = plt.figure()
    ax1 = fig.add_subplot(511, ylabel=' loss ')
    data['train/value_loss'].plot(ax=ax1, color='b', lw=1., legend=True)
    data['train/loss'].plot(ax=ax1, color='r', lw=1., legend=True)

    # sharex:设置同步缩放横轴，便于缩放查看
    ax2 = fig.add_subplot(512, ylabel='std', sharex=ax1)
    data['train/std'].plot(ax=ax2, color='g', lw=1., legend=True)

    ax3 = fig.add_subplot(513, ylabel='reward', sharex=ax1)
    data['train/reward'].plot(ax=ax3, color='b', lw=1., legend=True)
    data['train/clip_fraction'].plot(ax=ax3, color='black', lw=1., legend=True)

    ax4 = fig.add_subplot(514, ylabel='gr', sharex=ax1)
    data['train/approx_kl'].plot(ax=ax4, color='b', lw=1., legend=True)
    data['train/explained_variance'].plot(ax=ax4, color='r', lw=1., legend=True)
    data['train/policy_gradient_loss'].plot(ax=ax4, color='g', lw=1., legend=True)

    ax5 = fig.add_subplot(515, ylabel='en', sharex=ax1)
    data['train/entropy_loss'].plot(ax=ax5, color='r', lw=1., legend=True)

    plt.show()

    # 绘制


def back_test(trade, df_account_value, df_actions):
    ### Backtest
    time_list = cfg.time_list_get()
    tickers = cfg.ticker_list_get()
    # tickers = cfg.ticker_list_get()
    short_ticker = tickers[0].split(".", 1)[0]  # 临时，只取第一个
    print("ticker=", tickers, ",short_ticker=", short_ticker)

    plt.clf()
    plotter = ReturnPlotter(df_account_value, trade, time_list['trade_start_date'], time_list['trade_end_date'])
    plotter.plot_all()

    # plotter.plot_back(trade, df_actions, tickers)
    # plotter.plot_back_mul(trade, df_actions, tickers)

    plt.clf()
    # plotter.plot()

    # matplotlib inline
    # # ticket: SSE 50：000016
    plt.clf()
    plotter.plot(short_ticker)

    #### Use pyfolio

    # CSI 300  #沪深300  399300
    baseline_df = plotter.get_baseline(short_ticker)
    # print("baseline", baseline_df)

    daily_return = plotter.get_return(df_account_value)
    daily_return_base = plotter.get_return(baseline_df, value_col_name="close")

    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func(
        returns=daily_return,
        factor_returns=daily_return_base,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print("==============DRL Strategy Stats===========")
    print(f"perf_stats_all: {perf_stats_all}")

    daily_return = plotter.get_return(df_account_value)
    daily_return_base = plotter.get_return(baseline_df, value_col_name="close")

    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func(
        returns=daily_return_base,
        factor_returns=daily_return_base,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print("==============Baseline Strategy Stats===========")

    print(f"perf_stats_all: {perf_stats_all}")

    # plt_loss()


def get_returns(trade, df_account_value, df_actions):
    ### Backtest
    time_list = cfg.time_list_get()
    ticker = cfg.ticker_list_get()[0]
    short_ticker = ticker.split(".", 1)[0]
    print("ticker=", ticker, ",short_ticker=", short_ticker)

    plt.clf()
    plotter = ReturnPlotter(df_account_value, trade, time_list['trade_start_date'], time_list['trade_end_date'])
    # plotter.plot_all()

    # plotter.plot_back(trade, df_actions, ticker)

    # plt.clf()
    # plotter.plot()

    # matplotlib inline
    # # ticket: SSE 50：000016
    # plt.clf()
    # plotter.plot(short_ticker)

    #### Use pyfolio

    # CSI 300  #沪深300  399300
    baseline_df = plotter.get_baseline(short_ticker)
    # print("baseline", baseline_df)

    daily_return = plotter.get_return(df_account_value)
    daily_return_base = plotter.get_return(baseline_df, value_col_name="close")

    perf_func = timeseries.perf_stats
    perf_stats_all = perf_func(
        returns=daily_return,
        factor_returns=daily_return_base,
        positions=None,
        transactions=None,
        turnover_denom="AGB",
    )
    print("==============DRL Strategy Stats===========")
    print(f"perf_stats_all: {perf_stats_all}")
    return perf_stats_all
    #
    # daily_return = plotter.get_return(df_account_value)
    # daily_return_base = plotter.get_return(baseline_df, value_col_name="close")
    #
    # perf_func = timeseries.perf_stats
    # perf_stats_all = perf_func(
    #     returns=daily_return_base,
    #     factor_returns=daily_return_base,
    #     positions=None,
    #     transactions=None,
    #     turnover_denom="AGB",
    # )
    # print("==============Baseline Strategy Stats===========")
    #
    # print(f"perf_stats_all: {perf_stats_all}")


def all_run():
    time_list = cfg.time_list_get()
    p = data_process_creat(time_list['train_start_date'], time_list['trade_end_date'])  # all
    # p = data_process_creat(time_list['trade_start_date'], time_list['trade_end_date'])  # trade
    tickers = cfg.ticker_list_get()
    download_data(tickers + reference_tickers, p)

    add_technical_factors(p)



    new_logger = logconfigure("results/log", ["stdout", "csv", "tensorboard"])

    # ticker = cfg.ticker_list_get()[0].replace('.', '')
    res = '_'.join(map(lambda x: x.split('.')[0], tickers))
    print(res)

    model_name_a2c = 'train_a2c_' + res + '_0.zip'
    model_name_ddpg = 'train_ddpg_' + res + '_0.zip'
    model_name_td3 = 'train_td3_' + res + '_0.zip'
    model_name_sac = 'train_sac_' + res + '_0.zip'
    model_name_ppo = 'train_ppo_' + res + '_0.zip'

    # print("model name",model_name_a2c, ":",  model_name_ddpg)
    env = process_env(p, time_list['train_start_date'], time_list['train_end_date'])

    # agent, mod = agent_ddpg(env)
    # trained_model = data_train(agent, mod, model_name_ddpg)

    agent, mod = agent_a2c(env)
    mod.set_logger(new_logger)
    trained_model = data_train(agent, mod, model_name_a2c)

    # agent, mod = agent_td3(env)
    # trained_model = data_train(agent, mod, model_name_td3)

    # agent, mod = agent_sac(env)
    # trained_model = data_train(agent, mod, model_name_sac)

    # agent, mod = agent_ppo(env)
    # mod.set_logger(new_logger)
    # trained_model = data_train(agent, mod, model_name_ppo)

    # trained_model = load_model_file(mod, model_name_a2c)
    # trained_model = load_model_file(mod, model_name_ppo)

    # print(p.dataframe)
    trade, account_value, actions = data_predict(p, trained_model, time_list['trade_start_date'],
                                                 time_list['trade_end_date'])

    back_test(trade, account_value, actions)


def get_good_mode():
    time_list = cfg.time_list_get()
    p = data_process_creat(time_list['train_start_date'], time_list['trade_end_date'])  # all
    # p = data_process_creat(time_list['trade_start_date'], time_list['trade_end_date'])  # trade

    download_data(cfg.ticker_list_get() , p)
    add_technical_factors(p)

    ticker = cfg.ticker_list_get()[0].replace('.', '')
    model_name_a2c = 'train_a2c_' + ticker + '_0.zip'
    model_name_ddpg = 'train_ddpg_' + ticker + '_0.zip'
    model_name_td3 = 'train_td3_' + ticker + '_0.zip'
    model_name_sac = 'train_sac_' + ticker + '_0.zip'
    model_name_ppo = 'train_ppo_' + ticker + '_0.zip'

    # print("model name",model_name_a2c, ":",  model_name_ddpg)
    env = process_env(p, time_list['train_start_date'], time_list['train_end_date'])

    for i in range(1000):
        agent, mod = agent_a2c(env)
        trained_model = data_train(agent, mod, model_name_a2c)

        # agent, mod = agent_ddpg(env)
        # trained_model = data_train(agent, mod, model_name_ddpg)
        # agent, mod = agent_td3(env)
        # trained_model = data_train(agent, mod, model_name_td3)
        # agent, mod = agent_sac(env)
        # trained_model = data_train(agent, mod, model_name_sac)
        # agent, mod = agent_ppo(env)
        # trained_model = data_train(agent, mod, model_name_ppo)

        # trained_model = load_model_file(mod, model_name_ppo)
        trade, account_value, actions = data_predict(p, trained_model, time_list['trade_start_date'],
                                                     time_list['trade_end_date'])

        perf_stats = get_returns(trade, account_value, actions)
        annual = perf_stats['Annual return']
        cumulative = perf_stats['Cumulative returns']
        print("Annual and Cumulative return ", annual, cumulative)
        if annual >= 0.30:
            print("model is good, annual returns is ", annual)
            break
        else:
            print("annual is ", annual)


def get_good_mode_all():
    time_list = cfg.time_list_get()
    ticker_list_all = [['600111.SH'], ['000513.SZ'], ['600096.SH'], ['601111.SH']]
    for ticker_list in ticker_list_all:
        p = data_process_creat(time_list['train_start_date'], time_list['trade_end_date'])  # all
        download_data(ticker_list, p)
        add_technical_factors(p)

        ticker = ticker_list[0].replace('.', '')
        model_name_a2c = 'train_a2c_' + ticker + '_0.zip'
        model_name_ddpg = 'train_ddpg_' + ticker + '_0.zip'
        model_name_td3 = 'train_td3_' + ticker + '_0.zip'
        model_name_sac = 'train_sac_' + ticker + '_0.zip'
        model_name_ppo = 'train_ppo_' + ticker + '_0.zip'

        # print("model name",model_name_a2c, ":",  model_name_ddpg)
        env = process_env(p, time_list['train_start_date'], time_list['train_end_date'])

        for i in range(500):
            # agent, mod = agent_a2c(env)
            # trained_model = data_train(agent, mod, model_name_a2c)

            # agent, mod = agent_ddpg(env)
            # trained_model = data_train(agent, mod, model_name_ddpg)
            # agent, mod = agent_td3(env)
            # trained_model = data_train(agent, mod, model_name_td3)
            # agent, mod = agent_sac(env)
            # trained_model = data_train(agent, mod, model_name_sac)
            agent, mod = agent_ppo(env)
            trained_model = data_train(agent, mod, model_name_ppo)

            # trained_model = load_model_file(mod, model_name_ppo)
            trade, account_value, actions = data_predict(p, trained_model, time_list['trade_start_date'],
                                                         time_list['trade_end_date'])

            perf_stats = get_returns(trade, account_value, actions)
            annual = perf_stats['Annual return']
            cumulative = perf_stats['Cumulative returns']
            print("Annual and Cumulative return ", annual, cumulative)
            if annual >= 0.20:
                print("model is good , annual returns is ", annual)
                break
            else:
                print("annual is ", annual)


# get_good_mode_all()
# get_good_mode()
all_run()
# plt_loss()
