import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

import math

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

window_cnt = 20


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
            self,
            df,
            stock_dim,
            hmax,
            initial_amount,
            buy_cost_pct,
            sell_cost_pct,
            reward_scaling,
            state_space,
            action_space,
            tech_indicator_list,
            turbulence_threshold=None,
            make_plots=False,
            print_verbosity=2,
            day=0,
            initial=True,
            previous_state=[],
            model_name="",
            mode="",
            iteration="",
            initial_buy=False,  # Use half of initial amount to buy
            hundred_each_trade=True,
    ):  # The number of shares per lot must be an integer multiple of 100
        self.step_count = 0
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.initial_buy = initial_buy
        self.hundred_each_trade = hundred_each_trade
        self.window = window_cnt
        self.window_index = 0
        self.window_last_state = self._window_last_state()
        self.index = 0
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.reward_rate = self._incentive_rate(window_cnt)
        self.reward_windows_mean = []  # 每隔30天一个轮询，这30天的平均奖励值的所有记录

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []

        self.actions_memory_win = self._init_actions_memory_win()  # [[0 for i in range(window_cnt)] for j in range(window_cnt)]
        self.assert_memory_win = self._init_actions_memory_win()  # [[0 for i in range(window_cnt)] for j in range(window_cnt)]
        self.reward_memory_win = self._init_actions_memory_win()  # [[0 for i in range(window_cnt)] for j in range(window_cnt)]

        self.date_memory = [self._get_date()]
        self._seed()

    def _init_actions_memory_win(self):
        arr = [[0 for i in range(window_cnt)] for j in range(self.stock_dim)]
        # print(arr)

        return arr

    def _init_assert_memory_win(self):
        arr = [[0 for i in range(window_cnt)] for j in range(window_cnt)]
        return arr

    def _init_reward_memory_win(self):
        arr = [[0 for i in range(window_cnt)] for j in range(window_cnt)]
        return arr

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.state[index + 1] > 0:
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    if self.hundred_each_trade:
                        sell_num_shares = sell_num_shares // 100 * 100

                    sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct)
                    )
                    self.state[0] += sell_amount
                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                            self.state[index + 1] * sell_num_shares * self.sell_cost_pct
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            # print(self.turbulence, self.turbulence_threshold)
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                                self.state[index + 1]
                                * sell_num_shares
                                * (1 - self.sell_cost_pct)
                        )

                        self.state[0] += sell_amount

                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                                self.state[index + 1]
                                * self.state[index + self.stock_dim + 1]
                                * self.sell_cost_pct
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if self.state[index + 1] > 0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // self.state[index + 1]

                # update balance
                buy_num_shares = min(available_amount, action)
                if self.hundred_each_trade:
                    buy_num_shares = buy_num_shares // 100 * 100
                buy_amount = (
                        self.state[index + 1] * buy_num_shares * (1 + self.buy_cost_pct)
                )

                self.state[0] -= buy_amount
                self.state[index + self.stock_dim + 1] += buy_num_shares
                self.cost += self.state[index + 1] * buy_num_shares * self.buy_cost_pct
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            # print(self.turbulence, self.turbulence_threshold)
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def _incentive_rate(self, num):

        # 线性
        if num <= 0:
            return 0
        linear_array = []
        for i in range(num):
            # linear_array.append(i / num)
            linear_array.append(1)  # output 1
        # print(linear_array)
        return linear_array

        # # 对数
        # # print("************")
        # def log_growth_rate(x):
        #     return math.log(x + 1)
        #
        # x = [i / num for i in range(num)]
        # y = [log_growth_rate(i) for i in x]
        #
        # print("_incentive_rate x:",x, "y:",y)
        # return y

    def _calculate_reward_n(self, n):
        rates = self._incentive_rate(n)
        actions = self.actions_memory[-n:]
        assets = self.asset_memory[-n:]

        asset_cnt = sum((assets * rates))

        return 0

    def step(self, actions):
        lindex = len(self.df.index.unique())
        # print("step ...", self.day, len(self.df.index.unique()) - 1)

        # print("step actions:", actions)
        # print("step date_memory:", len(self.date_memory), "actions_memory:", len(self.actions_memory))
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                    self.state[0]
                    + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(
                    self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]
                )
            )
                    - self.initial_amount
            )
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                        (252 ** 0.5)
                        * df_total_value["daily_return"].mean()
                        / df_total_value["daily_return"].std()
                )

            print(self.rewards_memory)
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            # df_rewards["date"] = self.date_memory[:]

            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    f"results/actions_{self.mode}_{self.model_name}_{self.iteration}.csv"
                )
                df_total_value.to_csv(
                    f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.csv",
                    index=False,
                )
                df_rewards.to_csv(
                    f"results/account_rewards_{self.mode}_{self.model_name}_{self.iteration}.csv",
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.png",
                    index=False,
                )
                plt.close()

            return self.state, self.reward, self.terminal, {}

        else:
            self.step_count = self.step_count + 1
            # print(self.step_count, actions)
            # actions = np.where((abs(actions) < 0.8), 0, actions)

            # print("actions", actions)

            actions = actions * self.hmax  # actions initially is scaled between 0 to 1, 每次交易最大额度
            actions = actions.astype(
                int
            )  # convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:  # 震荡阈值
                # print(self.turbulence , self.turbulence_threshold)
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)

            actions_in = actions.copy()
            # print("-", actions_in)
            # begin_total_asset = self.initial_amount
            # begin_total_asset = self.state[0] + sum(
            #     np.array(self.state[1: (self.stock_dim + 1)])
            #     * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            # )

            begin_close = self.state[1: (self.stock_dim + 1)]
            begin_stock = self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])

            # print("+", actions)

            begin_total_asset = self._get_state_total_assert(self.window_last_state)
            self._update_window_last_state()
            end_total_asset = self._get_state_total_assert(self.window_last_state)
            # print(end_total_asset, begin_total_asset)

            # action_sub = np.array(actions_in) - np.array(actions)
            action_sub = abs(np.array(actions_in) - np.array(actions)) * -1  # 数值越大，说明买/卖数量偏差越大，则惩罚越大

            # print(actions_in, actions,"---" , action_sub)

            self.day += 1
            self.index += 1
            self.data = self.df.loc[self.day, :]
            self.state = self._update_state(action_sub)

            # if self.turbulence_threshold is not None:
            #     self.turbulence = self.data["turbulence"]
            #     # self.turbulence = self.data["turbulence"].values[0]

            i_list = []
            for i in range(self.stock_dim):
                if begin_stock[i] - self.state[self.stock_dim + 1 + i] == 0:
                    i_list.append(i)

            # end_total_asset = self.state[0] + sum(
            #     np.array(self.state[1: (self.stock_dim + 1)])
            #     * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
            # )

            self.reward = end_total_asset - begin_total_asset  # + action_sub * begin_close * 0.1

            # print("reward:", self.reward, end_total_asset - begin_total_asset, action_sub * begin_close * 0.1)

            for i in i_list:
                self.reward -= (
                        self.state[i + 1] * self.state[self.stock_dim + 1 + i] * 0.01
                )
            self.rewards_memory.append(self.reward)

            self.reward = self.reward * self.reward_scaling * self.reward_rate[self.index - 1]

            # print(self.day, self._get_date(), begin_total_asset, end_total_asset, "--", actions,
            #       self.state, self.window_last_state[:-4], "reward", self.reward)

            # print("sell_index", sell_index, "    buy_index", buy_index)
            # print(self.state, self.reward, self.terminal)
            # print(self.rewards_memory)

            self.actions_memory.append(actions)  #
            self.asset_memory.append(end_total_asset)  #
            self.date_memory.append(self._get_date())

            if self.index >= self.window:
                # 计算平均收益
                window_reward = sum(self.rewards_memory)
                self.reward_windows_mean.append(window_reward / (self.index + 1))

                self.index = 0
                self.day -= (self.window - 1)
                self._window_reset()

                self.rewards_memory.clear()  # 清空奖励数组,以reward_windows_mean记录作为最后统计
                self.actions_memory.clear()
                self.asset_memory = [self.initial_amount]
                self.date_memory = [self._get_date()]

                # self.actions_memory_win[self.window_index].append()

                self.window_index += 1

                # print("self.day", self.day, self._get_date(), "asset", end_total_asset, actions,
                #       self.state[1: (self.stock_dim + 1)])

            return self.state, self.reward, self.terminal, {}

    def _window_reset(self):
        self.data = self.df.loc[self.day, :]  # todo

        # initiate state
        self.state = self._reset_state()
        self.window_last_state = self._window_last_state()

        # self.turbulence = 0
        # self.cost = 0
        # self.trades = 0
        # self.terminal = False
        # self.iteration=self.iteration
        # self.rewards_memory = []
        # self.actions_memory = []
        # self.date_memory = [self._get_date()]

        # self.episode += 1
        #
        # if self.initial:
        #     self.asset_memory = [self.initial_amount]
        # else:
        #     previous_total_asset = self.previous_state[0] + sum(
        #         np.array(self.state[1: (self.stock_dim + 1)])
        #         * np.array(
        #             self.previous_state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]
        #         )
        #     )
        #     self.asset_memory = [previous_total_asset]

        return self.state

    def reset(self):
        # initiate state
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [self.initial_amount]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1: (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]
        self.day = 0
        self.index = 0
        self.data = self.df.loc[self.day, :]
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                        [self.initial_amount]  # amount  1000000
                        + self.data.close.values.tolist()  # close price
                        + [0] * self.stock_dim  # buy/sell count
                        + sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [], )  # tech
                        + [0]  # index in range [0:30]
                        + [0] * self.stock_dim  # 提供的action 与实际操作时的差值
                )

                if self.initial_buy:
                    state = self.initial_buy_()
            else:
                # for single stock
                state = (
                        [self.initial_amount]
                        + [self.data.close]
                        + [0] * self.stock_dim
                        + sum([[self.data[tech]] for tech in self.tech_indicator_list], [], )
                        + [0]  # index in range [0:30]
                        + [0] * self.stock_dim  # 提供的action 与实际操作时的差值
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                        [self.previous_state[0]]
                        + self.data.close.values.tolist()
                        + self.previous_state[
                          (self.stock_dim + 1): (self.stock_dim * 2 + 1)
                          ]
                        + sum([self.data[tech].values.tolist()
                               for tech in self.tech_indicator_list], [], )
                        + [0]  # index in range [0:30]
                        + [0] * self.stock_dim  # 提供的action 与实际操作时的差值
                )
            else:
                # for single stock
                state = (
                        [self.previous_state[0]]
                        + [self.data.close]
                        + self.previous_state[
                          (self.stock_dim + 1): (self.stock_dim * 2 + 1)
                          ]
                        + sum([[self.data[tech]] for tech in self.tech_indicator_list], [], )
                        + [0]  # index in range [0:30]
                        + [0] * self.stock_dim  # 提供的action 与实际操作时的差值
                )
        return state

    def _window_last_state(self):

        print(self.df.index)

        last_data = []
        if (len(self.df) - self.day) < self.window:
            last_data = self.df.loc[self.df.index[-1]]  # 取最后一个
        else:
            last_data = self.df.loc[self.df.index[self.day + self.window], :]  # 多只股票？
            # last_data = self.df.loc[self.df.index[self.day + self.window]:, :]  #

        print(last_data)

        # # 这里需要判断是否超出范围！
        # last_data = []
        # if (len(self.df) - self.day) < self.window:
        #     last_data = self.df.loc[self.df.index[-1]]  # 取最后一个
        # else:
        #     last_data = self.df.loc[self.day + self.window - 1:, :]

        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                        [self.initial_amount]  # amount  1000000
                        + last_data.close.values.tolist()  # close price
                        + [0] * self.stock_dim  # buy/sell count
                        + sum([last_data[tech].values.tolist() for tech in self.tech_indicator_list], [], )  # tech
                        + [0]  # index in range [0:30]
                        + [0] * self.stock_dim  # 提供的action 与实际操作时的差值
                )

                if self.initial_buy:
                    state = self.initial_buy_()
            else:
                # for single stock
                state = (
                        [self.initial_amount]
                        + [last_data.close]
                        + [0] * self.stock_dim
                        + sum([[last_data[tech]] for tech in self.tech_indicator_list], [], )
                        + [0]  # index in range [0:30]
                        + [0] * self.stock_dim  # 提供的action 与实际操作时的差值
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                        [self.previous_state[0]]
                        + last_data.close.values.tolist()
                        + self.previous_state[
                          (self.stock_dim + 1): (self.stock_dim * 2 + 1)
                          ]
                        + sum([last_data[tech].values.tolist()
                               for tech in self.tech_indicator_list], [], )
                        + [0]  # index in range [0:30]
                        + [0] * self.stock_dim  # 提供的action 与实际操作时的差值
                )
            else:
                # for single stock
                state = (
                        [self.previous_state[0]]
                        + [last_data.close]
                        + self.previous_state[
                          (self.stock_dim + 1): (self.stock_dim * 2 + 1)
                          ]
                        + sum([[last_data[tech]] for tech in self.tech_indicator_list], [], )
                        + [0]  # index in range [0:30]
                        + [0] * self.stock_dim  # 提供的action 与实际操作时的差值
                )
        return state

    def _reset_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + [0] * self.stock_dim
                    + sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [], )
                    + [0]  # index
                    + [0] * self.stock_dim  # 提供的action 与实际操作时的差值
            )

        else:
            # for single stock
            state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim  # buy/sell count
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], [], )
                    + [0]  # index
                    + [0] * self.stock_dim  # 提供的action 与实际操作时的差值
            )
        return state

    def _update_state(self, actions):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                    [self.state[0]]
                    + self.data.close.values.tolist()
                    + list(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
                    + sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [], )
                    + [self.index]
                    + list(actions)  # [?]提供的action 与实际操作时的差值
            )

        else:
            # for single stock
            state = (
                    [self.state[0]]
                    + [self.data.close]
                    + list(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])
                    + sum([[self.data[tech]] for tech in self.tech_indicator_list], [], )
                    + [self.index]
                    + list(actions)  # + [0]* self.stock_dim  # [?]提供的action 与实际操作时的差值
            )

        # print(state)
        # if state[self.stock_dim + 1] != 0:
        #     print("++++++++++++")

        return state

    def _update_window_last_state(self):
        # 将当前窗口下的股票数量、现金额，同步到窗口期的最后，用于统计和作为股票涨跌的参考标的
        self.window_last_state = (
                [self.state[0]]  # 现金
                + list(self.window_last_state[1: (self.stock_dim + 1)])  # 价格
                + list(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])  # 股票数量
                + list(self.window_last_state[(self.stock_dim * 2 + 1):])  # tech + index + actions
            # + [self.index]
            # + list(actions)  # [?]提供的action 与实际操作时的差值
        )

    def _get_state_total_assert(self, state):
        total_asset = state[0] + sum(
            np.array(state[1: (self.stock_dim + 1)])  # 价格
            * np.array(state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])  # 数量
        )
        return total_asset

    def _get_date(self):

        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def _total_asset(self):

        total_asset = self.state[0] + sum(
            np.array(self.state[1: (self.stock_dim + 1)])  # price
            * np.array(self.state[(self.stock_dim + 1): (self.stock_dim * 2 + 1)])  # buy/sell count
        )
        return total_asset

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory

        print("date_list", len(date_list), date_list)
        print("asset_list", len(asset_list), asset_list)

        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):

        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            # df_actions.index = df_date.date[2:]
            date_list = date_list[1:]
            df_actions = pd.DataFrame({'date': date_list, 'actions': action_list})
            return df_actions
        else:
            # date_list = self.date_memory[:]
            # date_list = self.date_memory[:-1]
            date_list = self.date_memory[1:]
            action_list = self.actions_memory

            print("date_list", len(date_list), "actions_memory", len(action_list))
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
            return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def initial_buy_(self):
        """Initialize the state, already bought some"""

        prices = self.data.close.values.tolist()
        avg_price = sum(prices) / len(prices)
        buy_nums_each_tic = (
                0.5 * self.initial_amount // (avg_price * len(prices))
        )  # only use half of the initial amount
        cost = sum(prices) * buy_nums_each_tic

        state = (
                [self.initial_amount - cost]
                + self.data.close.values.tolist()
                + [buy_nums_each_tic] * self.stock_dim
                + sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [], )
                + [0]
                + [0] * self.stock_dim  # 提供的action 与实际操作时的差值
        )

        return state
