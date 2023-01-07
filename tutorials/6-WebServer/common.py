import os
from typing import List
import config_parse as cfg
# from meta.data_processor import DataProcessor

print("common....")

# "./" will be added in front of each directory
def check_and_make_directories(directories: List[str]):
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)

def cache_file(start, end):
    ticker_list = cfg.ticker_list_get()
    time_list = cfg.time_list_get()
    data_source = cfg.curent_data_source_get()
    dir_list = cfg.dir_list_get()

    cache_filename = (
            "_".join(
                ticker_list
                + [
                    data_source,
                    # time_list['train_start_date'],
                    # time_list['trade_end_date'],
                    start,
                    end,
                    time_list['time_interval'],
                ]
            )
            + ".pickle"
    )
    cache_dir = dir_list['data_save_dir']  # "./cache/"
    cache_path = os.path.join(cache_dir, cache_filename)
    return cache_path, cache_dir, cache_filename


def data_process(p, start, end, del_cnt=0):
    # time_list = cfg.time_list_get()
    # 丢弃前n个数据，避免数据为0影响训练结果
    if del_cnt != 0:
        p.dataframe = p.dataframe.iloc[20:, :].reset_index(drop=True)
        # print(f"p.dataframe: {p.dataframe}")

    # train = p.data_split(p.dataframe, time_list['train_start_date'], time_list['train_end_date'])
    train = p.data_split(p.dataframe, start, end)
    # print(f"len(train.tic.unique()): {len(train.tic.unique())}")

    print(f"train.tic.unique(): {train.tic.unique()}")
    print(f"train.head(): {train.head()}")
    print(f"train.tail(): {train.tail()}")
    print(f"train.shape: {train.shape}")
    return train

def env_kwargs(data):
    stock_dimension = len(data.tic.unique())
    state_space = stock_dimension * (len(cfg.indicators_get()) + 2) + 1
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    # kwargs = {
    #     "stock_dim": stock_dimension,
    #     "hmax": 10000,
    #     "initial_amount": 1000000,
    #     "buy_cost_pct": 6.87e-5,
    #     "sell_cost_pct": 1.0687e-3,
    #     "reward_scaling": 1e-4,
    #     "state_space": state_space,
    #     "action_space": stock_dimension,
    #     "tech_indicator_list": cfg.indicators_get(),
    #     "print_verbosity": 1,
    #     "initial_buy": False,
    #     "hundred_each_trade": True,
    # }
    kwargs = {
        "stock_dim": stock_dimension,
        "hmax": 10000,
        "initial_amount": 1000000,
        # "buy_cost_pct": 3.000e-4,
        # "sell_cost_pct": 2.250e-3,
        # "reward_scaling": 2.000e-4,
        "buy_cost_pct": 6.87e-5,
        "sell_cost_pct": 1.0687e-3,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": stock_dimension,
        "tech_indicator_list": cfg.indicators_get(),
        "print_verbosity": 1,
        "initial_buy": False,
        "hundred_each_trade": True,
    }
    return kwargs




