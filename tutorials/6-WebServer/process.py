import os
import sys

sys.path.append("../../")

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

from meta import config
import pyfolio
from pyfolio import timeseries
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline
from IPython import display

# 系统参数配置
print(matplotlib.get_backend())
matplotlib.use('TkAgg')  # TkAgg  'module://backend_interagg'
warnings.filterwarnings("ignore")
pd.options.display.max_columns = None

# 目录结构创建
dirs = cfg.dir_list_get()
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
    file_path, file_dir, file_name = common.cache_file(p.start_date, p.end_date)
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

    p.add_technical_indicator(indicator)
    p.add_technical_factor(factor_list)

    p.clean_data()
    print(f"p.dataframe: {p.dataframe}")


### Train
def process_env(p, start, end):
    time_list = cfg.time_list_get()
    indicators = cfg.indicators_get()

    # train = common.data_process(p, time_list['train_start_date'], time_list['train_end_date'], 0)
    train = common.data_process(p, start, end, 0)
    stock_dimension = len(train.tic.unique())
    state_space = stock_dimension * (len(indicators) + 2) + 1
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

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


def data_train(agent, model, name):
    trained_model = agent.train_model(
        model=model, tb_log_name=name, total_timesteps=10000
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

    model_file =f"{dir_list['trained_model_dir']}/{name}"
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
    trade = p.data_split(p.dataframe, start, end)
    print("trade.......")
    print(trade)
    env_kwargs = common.env_kwargs(trade)
    e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)

    # df_account_value, df_actions = DRLAgent.DRL_prediction(
    #     model=trained_ddpg, environment=e_trade_gym
    # )

    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=model, environment=e_trade_gym
    )
    print(df_account_value)
    print(df_actions)

    return trade, df_account_value, df_actions


def back_test(trade, df_account_value, df_actions):
    ### Backtest
    time_list = cfg.time_list_get()

    plt.clf()
    plotter = ReturnPlotter(df_account_value, trade, time_list['trade_start_date'], time_list['trade_end_date'])
    plotter.plot_all()

    plotter.plot_back(trade, df_actions, '600009.SH')

    plt.clf()
    plotter.plot()

    # matplotlib inline
    # # ticket: SSE 50：000016
    plt.clf()
    plotter.plot("000016")

    #### Use pyfolio

    # CSI 300  #沪深300  399300
    baseline_df = plotter.get_baseline("600009")

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


def test_run():
    time_list = cfg.time_list_get()
    p = data_process_creat(time_list['train_start_date'], time_list['trade_end_date'])  # all
    # p = data_process_creat(time_list['trade_start_date'], time_list['trade_end_date'])  # trade

    download_data(cfg.ticker_list_get(), p)
    add_technical_factor(p)

    env = process_env(p, time_list['train_start_date'], time_list['train_end_date'])
    # agent, mod = agent_ddpg(env)
    agent, mod = agent_a2c(env)
    # trained_model = data_train(agent, mod, "train")

    trained_model = load_model_file(mod, 'train_10k_0.zip')

    trade, account_value, actions = data_predict(p, trained_model, time_list['trade_start_date'],
                                                 time_list['trade_end_date'])

    back_test(trade, account_value, actions)

    exit()
