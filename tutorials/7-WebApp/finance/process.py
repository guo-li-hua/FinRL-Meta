import os
import sys

sys.path.append("../../../")
sys.path.append("../../../../FinRL-Meta")
sys.path.append("./finance")

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
from meta.env_stock_trading.env_stocktrading_China_A_shares_30days import (
    StockTradingEnv,
)

from agents.stablebaselines3_models import DRLAgent

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


def add_technical_factor(p):
    indicator = cfg.indicators_get()
    factor_list = cfg.factors_get()

    # p.clean_data()
    # p.add_turbulence()
    # print("before add_technical_indicator", p.dataframe.shape)
    p.add_technical_indicator(indicator)
    # print("after add_technical_indicator", p.dataframe.shape)

    p.add_technical_factor(factor_list)
    # print("after add_technical_factor", p.dataframe.shape)
    print(p.dataframe)

    p.clean_data()

    # print(f"p.dataframe: {p.dataframe}")


### Train
def process_env(p, start, end):
    time_list = cfg.time_list_get()
    indicators = cfg.indicators_get()
    factors = cfg.factors_get()

    # train = common.data_process(p, time_list['train_start_date'], time_list['train_end_date'], 0)
    train = common.data_process(p, start, end, 0)
    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension * (len(indicators) + len(factors) + 2) + 1
    # print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = common.env_kwargs(train)
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    print(f"print(type(env_train)): {print(type(env_train))}")
    return env_train


def agent_ddpg(env_train):
    ## DDPG
    agent = DRLAgent(env=env_train)
    DDPG_PARAMS = {
        "batch_size": 256,
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
        model=model, tb_log_name=name, total_timesteps=7000
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


def back_test(trade, df_account_value, df_actions):
    ### Backtest
    time_list = cfg.time_list_get()
    ticker = cfg.ticker_list_get()[0]
    short_ticker = ticker.split(".", 1)[0]
    print("ticker=", ticker, ",short_ticker=", short_ticker)

    plt.clf()
    plotter = ReturnPlotter(df_account_value, trade, time_list['trade_start_date'], time_list['trade_end_date'])
    plotter.plot_all()

    plotter.plot_back(trade, df_actions, ticker)

    plt.clf()
    plotter.plot()

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

    download_data(cfg.ticker_list_get(), p)
    add_technical_factor(p)

    ticker = cfg.ticker_list_get()[0].replace('.', '')
    model_name_a2c = 'train_a2c_' + ticker + '_0.zip'
    model_name_ddpg = 'train_ddpg_' + ticker + '_0.zip'
    model_name_td3 = 'train_td3_' + ticker + '_0.zip'
    model_name_sac = 'train_sac_' + ticker + '_0.zip'
    model_name_ppo = 'train_ppo_' + ticker + '_0.zip'

    # print("model name",model_name_a2c, ":",  model_name_ddpg)
    env = process_env(p, time_list['train_start_date'], time_list['train_end_date'])

    # agent, mod = agent_ddpg(env)
    # trained_model = data_train(agent, mod, model_name_ddpg)

    agent, mod = agent_a2c(env)
    trained_model = data_train(agent, mod, model_name_a2c)

    # agent, mod = agent_td3(env)
    # trained_model = data_train(agent, mod, model_name_td3)

    # agent, mod = agent_sac(env)
    # trained_model = data_train(agent, mod, model_name_sac)

    # agent, mod = agent_ppo(env)
    # trained_model = data_train(agent, mod, model_name_ppo)

    # trained_model = load_model_file(mod, model_name_a2c)

    # print(p.dataframe)
    trade, account_value, actions = data_predict(p, trained_model, time_list['trade_start_date'],
                                                 time_list['trade_end_date'])

    back_test(trade, account_value, actions)


def get_good_mode():
    time_list = cfg.time_list_get()
    p = data_process_creat(time_list['train_start_date'], time_list['trade_end_date'])  # all
    # p = data_process_creat(time_list['trade_start_date'], time_list['trade_end_date'])  # trade

    download_data(cfg.ticker_list_get(), p)
    add_technical_factor(p)

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
        add_technical_factor(p)

        ticker = ticker_list[0].replace('.', '')
        model_name_a2c = 'train_a2c_' + ticker + '_0.zip'
        model_name_ddpg = 'train_ddpg_' + ticker + '_0.zip'
        model_name_td3 = 'train_td3_' + ticker + '_0.zip'
        model_name_sac = 'train_sac_' + ticker + '_0.zip'
        model_name_ppo = 'train_ppo_' + ticker + '_0.zip'

        # print("model name",model_name_a2c, ":",  model_name_ddpg)
        env = process_env(p, time_list['train_start_date'], time_list['train_end_date'])

        for i in range(500):
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
            if annual >= 0.20:
                print("model is good , annual returns is ", annual)
                break
            else:
                print("annual is ", annual)

# get_good_mode_all()
# get_good_mode()
# all_run()
