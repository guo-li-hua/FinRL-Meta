import matplotlib

print(matplotlib.get_backend())
matplotlib.use('TkAgg')  # TkAgg  'module://backend_interagg'
import matplotlib.pyplot as plt
# import numpy as np
# x = np.arange(1,10)
# y = x
# plt.plot(x, y, "o:r")
# plt.show()
# exit()


import datetime

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from IPython import display

# display.set_matplotlib_formats("svg")
# display.display_svg("svg")


from meta.data_processor import DataProcessor
from main import check_and_make_directories
from meta.data_processors.tushare import Tushare, ReturnPlotter
from meta.env_stock_trading.env_stocktrading_China_A_shares_30days import (
    StockTradingEnv,
)
from agents.stablebaselines3_models import DRLAgent
import os
from typing import List
from argparse import ArgumentParser
from meta import config
from meta.config_tickers import DOW_30_TICKER
from meta.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
    ERL_PARAMS,
    RLlib_PARAMS,
    SAC_PARAMS,
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_API_BASE_URL,
)
import pyfolio
from pyfolio import timeseries

from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

pd.options.display.max_columns = None

print("ALL Modules have been imported!")

### Create folders

import os

"""
use check_and_make_directories() to replace the following

if not os.path.exists("./datasets"):
    os.makedirs("./datasets")
if not os.path.exists("./trained_models"):
    os.makedirs("./trained_models")
if not os.path.exists("./tensorboard_log"):
    os.makedirs("./tensorboard_log")
if not os.path.exists("./results"):
    os.makedirs("./results")
"""

check_and_make_directories(
    [DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR]
)

### Download data, cleaning and feature engineering

ticker_list = [
    # "510800.SH",
    # "600000.SH",
    # "BTC-USDT",
    "600009.SH",
    # "600016.SH",
    # "600028.SH",
    # "600030.SH",
    # "600031.SH",
    # "600036.SH",
    # "600050.SH",
    # "600104.SH",
    # "600196.SH",
    # "600276.SH",
    # "600309.SH",
    # "600519.SH",
    # "600547.SH",
    # "600570.SH",
]
# At Oct.22 2022, trade date available span is [2020-04-22, 2022-10-21]
TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2022-10-06"
TRADE_START_DATE = "2022-10-07"
TRADE_END_DATE = "2022-11-07"

# 1个月  3个月
TIME_INTERVAL = "1d"
kwargs = {}
kwargs["token"] = "3082a6ea9b417fdfb3d32e4af9cde1e7af610246378c131ef24ab062"
# "27080ec403c0218f96f388bca1b1d85329d563c91a43672239619ef5"
p = DataProcessor(
    data_source="tushare",  # "tushare",
    start_date=TRAIN_START_DATE,
    end_date=TRADE_END_DATE,
    time_interval=TIME_INTERVAL,
    **kwargs,
)

# download and clean
# p.download_data(ticker_list=ticker_list)
price_array, tech_array, turbulence_array = p.run(ticker_list, config.INDICATORS, False, cache=True)

p.clean_data()
# print(p.dataframe)

# add_technical_indicator
# p.add_technical_indicator(config.INDICATORS)
# p.clean_data()

# 添加因子
factor_list = [
    "bias_5_days",
    "roc_6_days",
    "vstd_10_days",
]
p.add_technical_factor(factor_list)
# print(f"p.dataframe: {p.dataframe}")

# 丢弃前n个数据，避免数据为0影响训练结果
p.dataframe = p.dataframe.iloc[20:, :].reset_index(drop=True)
print(f"p.dataframe: {p.dataframe}")

### Split traning dataset

train = p.data_split(p.dataframe, TRAIN_START_DATE, TRAIN_END_DATE)
print(f"len(train.tic.unique()): {len(train.tic.unique())}")

print(f"train.tic.unique(): {train.tic.unique()}")

print(f"train.head(): {train.head()}")

print(f"train.shape: {train.shape}")

stock_dimension = len(train.tic.unique())
state_space = stock_dimension * (len(config.INDICATORS) + 2) + 1
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

### Train

env_kwargs = {
    "stock_dim": stock_dimension,
    "hmax": 1000,
    "initial_amount": 1000000,
    "buy_cost_pct": 6.87e-5,
    "sell_cost_pct": 1.0687e-3,
    "reward_scaling": 1e-4,
    "state_space": state_space,
    "action_space": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "print_verbosity": 1,
    "initial_buy": True,
    "hundred_each_trade": True,
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)

## DDPG

env_train, _ = e_train_gym.get_sb_env()
print(f"print(type(env_train)): {print(type(env_train))}")

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

# trained_ddpg = agent.train_model(
#     model=model_ddpg, tb_log_name="ddpg", total_timesteps=10000
# )

## A2C

agent = DRLAgent(env=env_train)
model_a2c = agent.get_model("a2c")

# trained_a2c = agent.train_model(
#     model=model_a2c, tb_log_name="a2c", total_timesteps=50000
# )
model_file_name = "a2c_50k_0.zip"
trained_a2c_read = model_a2c.load(f"{config.TRAINED_MODEL_DIR}/{model_file_name}")

### Trade
trade = p.data_split(p.dataframe, TRADE_START_DATE, TRADE_END_DATE)
env_kwargs = {
    "stock_dim": stock_dimension,
    "hmax": 10000,
    "initial_amount": 1000000,
    "buy_cost_pct": 6.87e-5,
    "sell_cost_pct": 1.0687e-3,
    "reward_scaling": 1e-4,
    "state_space": state_space,
    "action_space": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "print_verbosity": 1,
    "initial_buy": False,
    "hundred_each_trade": True,
}
e_trade_gym = StockTradingEnv(df=trade, **env_kwargs)

# df_account_value, df_actions = DRLAgent.DRL_prediction(
#     model=trained_ddpg, environment=e_trade_gym
# )


df_account_value, df_actions = DRLAgent.DRL_prediction(
    model=trained_a2c_read, environment=e_trade_gym
)
print(df_account_value)
print(df_actions)

df_actions.to_csv("action.csv", index=False)
print(f"df_account_value: {df_account_value}")
print(f"df_actions: {df_actions}")


# DRL_prediction_load_from_file


### Backtest

plt.clf()
plotter = ReturnPlotter(df_account_value, trade, TRADE_START_DATE, TRADE_END_DATE)
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

exit()

# Part 7: Backtesting Results
print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)
perf_stats_all.to_csv("./" + RESULTS_DIR + "/perf_stats_all_" + now + '.csv')

# baseline stats
print("==============Get Baseline Stats===========")
# baseline_df = get_baseline(
#     ticker="600009",
#     start=df_account_value.loc[0, 'date'],
#     end=df_account_value.loc[len(df_account_value) - 1, 'date'])

stats = backtest_stats(baseline_df, value_col_name='close')

print(df_account_value.loc[0, 'date'])
print(df_account_value.loc[len(df_account_value) - 1, 'date'])

## 7.2 BackTestPlot

print("==============Compare to DJIA===========")

# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(df_account_value,
              baseline_ticker='600009',
              baseline_start=df_account_value.loc[0, 'date'],
              baseline_end=df_account_value.loc[len(df_account_value) - 1, 'date'])
