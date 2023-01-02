import os
import pickle
from typing import List

import numpy as np
import pandas as pd

from meta.factors.factors import MomentumFactors as monentum
from meta.factors.factors import EmotionFactors as emotion
from meta.factors.factors import extraFacters as extra
from meta.factors.factors import generalFactors as general


class DataProcessor:
    def __init__(
            self,
            data_source: str,
            start_date: str,
            end_date: str,
            time_interval: str,
            **kwargs,
    ):
        self.data_source = data_source
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval
        self.dataframe = pd.DataFrame()

        if self.data_source == "alpaca":
            from meta.data_processors.alpaca import Alpaca

            processor_dict = {self.data_source: Alpaca}

        elif self.data_source == "alphavantage":
            from meta.data_processors.alphavantage import Alphavantage

            processor_dict = {self.data_source: Alphavantage}

        elif self.data_source == "baostock":
            from meta.data_processors.baostock import Baostock

            processor_dict = {self.data_source: Baostock}

        elif self.data_source == "binance":
            from meta.data_processors.binance import Binance

            processor_dict = {self.data_source: Binance}

        elif self.data_source == "binanceTu":
            from meta.data_processors.binancetu import Binancetu

            processor_dict = {self.data_source: Binancetu}
        elif self.data_source == "ccxt":
            from meta.data_processors.ccxt import Ccxt

            processor_dict = {self.data_source: Ccxt}

        elif self.data_source == "iexcloud":
            from meta.data_processors.iexcloud import Iexcloud

            processor_dict = {self.data_source: Iexcloud}

        elif self.data_source == "joinquant":
            from meta.data_processors.joinquant import Joinquant

            processor_dict = {self.data_source: Joinquant}

        elif self.data_source == "quandl":
            from meta.data_processors.quandl import Quandl

            processor_dict = {self.data_source: Quandl}

        elif self.data_source == "quantconnect":
            from meta.data_processors.quantconnect import Quantconnect

            processor_dict = {self.data_source: Quantconnect}

        elif self.data_source == "ricequant":
            from meta.data_processors.ricequant import Ricequant

            processor_dict = {self.data_source: Ricequant}

        elif self.data_source == "tushare":
            from meta.data_processors.tushare import Tushare

            processor_dict = {self.data_source: Tushare}

        elif self.data_source == "wrds":
            from meta.data_processors.wrds import Wrds

            processor_dict = {self.data_source: Wrds}

        elif self.data_source == "yahoofinance":
            from meta.data_processors.yahoofinance import Yahoofinance

            processor_dict = {self.data_source: Yahoofinance}

        else:
            print(f"{self.data_source} is NOT supported yet.")

        try:
            self.processor = processor_dict.get(self.data_source)(
                data_source, start_date, end_date, time_interval, **kwargs
            )
            print(f"{self.data_source} successfully connected")
        except:
            raise ValueError(
                f"Please input correct account info for {self.data_source}!"
            )

    def download_data(self, ticker_list):
        self.processor.download_data(ticker_list=ticker_list)
        self.dataframe = self.processor.dataframe

    def clean_data(self):
        self.processor.dataframe = self.dataframe
        self.processor.clean_data()
        self.dataframe = self.processor.dataframe

    def add_technical_indicator(
            self, tech_indicator_list: List[str], select_stockstats_talib: int = 0
    ):
        self.tech_indicator_list = tech_indicator_list
        self.processor.add_technical_indicator(
            tech_indicator_list, select_stockstats_talib
        )
        self.dataframe = self.processor.dataframe

    def add_technical_factor(self, tech_factor_list: List[str]):
        for fact in tech_factor_list:
            if fact == "fft":  # 傅里叶
                self.processor.add_technical_factor_fft()
            # monentum
            elif fact == "bias_5_days":
                self.dataframe['bias_5_days'] = monentum.bias_5_days(self.dataframe['close'])
            elif fact == "roc_6_days":
                self.dataframe['roc_6_days'] = monentum.roc_6_days(self.dataframe['close'])
            # emotion
            elif fact == "vstd_10_days":
                self.dataframe['vstd_10_days'] = emotion.vstd_10_days(self.dataframe['volume'])
            elif fact == "vosc":
                self.dataframe['vosc'] = emotion.vosc(self.dataframe['volume'])

        self.dataframe = self.processor.dataframe

    def add_technical_factor_with_data(self, df, tech_factor_list: List[str]):
        final_df = pd.DataFrame()

        for i in df.tic.unique():
            tic_df = df[df.tic == i].copy()
            for fact in tech_factor_list:
                # if fact == "fft":  # 傅里叶
                #     self.processor.add_technical_factor_fft()
                # monentum
                if fact == "bias_5_days":
                    tic_df['bias_5_days'] = monentum.bias_5_days(tic_df['close'])
                elif fact == "bias_10_days":
                    tic_df['bias_10_days'] = monentum.bias_10_days(tic_df['close'])
                elif fact == "bias_60_days":
                    tic_df['bias_60_days'] = monentum.bias_60_days(tic_df['close'])
                elif fact == "price_1_month":
                    tic_df['price_1_month'] = monentum.price_1_month(tic_df['close'])
                elif fact == "price_3_monthes":
                    tic_df['price_3_monthes'] = monentum.price_3_monthes(tic_df['close'])
                elif fact == "roc_6_days":
                    tic_df['roc_6_days'] = monentum.roc_6_days(tic_df['close'])
                elif fact == "roc_12_days":
                    tic_df['roc_12_days'] = monentum.roc_12_days(tic_df['close'])
                elif fact == "roc_20_days":
                    tic_df['roc_20_days'] = monentum.roc_20_days(tic_df['close'])
                elif fact == "single_day_vpt":
                    tic_df['single_day_vpt'] = monentum.single_day_vpt(tic_df)
                elif fact == "single_day_vpt_6":
                    tic_df['single_day_vpt_6'] = monentum.single_day_vpt_6(tic_df)
                elif fact == "single_day_vpt_12":
                    tic_df['single_day_vpt_12'] = monentum.single_day_vpt_12(tic_df)
                elif fact == "cci_10_days":
                    tic_df['cci_10_days'] = monentum.cci_10_days(tic_df)
                elif fact == "cci_15_days":
                    tic_df['cci_15_days'] = monentum.cci_15_days(tic_df)
                elif fact == "cci_20_days":
                    tic_df['cci_20_days'] = monentum.cci_20_days(tic_df)
                elif fact == "bull_power":
                    tic_df['bull_power'] = monentum.bull_power(tic_df)
                # emotion
                elif fact == "vstd_10_days":
                    tic_df['vstd_10_days'] = emotion.vstd_10_days(tic_df['volume'])
                elif fact == "vstd_20_days":
                    tic_df['vstd_20_days'] = emotion.vstd_20_days(tic_df['volume'])
                elif fact == "tvstd_6_days":
                    tic_df['tvstd_6_days'] = emotion.tvstd_6_days(tic_df)
                elif fact == "tvstd_20_days":
                    tic_df['tvstd_20_days'] = emotion.tvstd_20_days(tic_df)
                elif fact == "vema_5_days":
                    tic_df['vema_5_days'] = emotion.vema_5_days(tic_df['volume'])
                elif fact == "vema_10_days":
                    tic_df['vema_10_days'] = emotion.vema_10_days(tic_df['volume'])
                elif fact == "vosc":
                    tic_df['vosc'] = emotion.vosc(tic_df['volume'])
                elif fact == "vroc_6_days":
                    tic_df['vroc_6_days'] = emotion.vroc_6_days(tic_df['volume'])
                elif fact == "vroc_12_days":
                    tic_df['vroc_12_days'] = emotion.vroc_12_days(tic_df['volume'])
                elif fact == "tvma_6_days":
                    tic_df['tvma_6_days'] = emotion.tvma_6_days(tic_df)
                elif fact == "wvad":
                    tic_df['wvad'] = emotion.wvad(tic_df)
                elif fact == "ar":
                    tic_df['ar'] = emotion.ar(tic_df)
                # extraFacters
                elif fact == "rsrs":
                    tic_df['rsrs'] = extra.rsrs(tic_df, 10)
                # generalFactors
                elif fact == "macd":
                    tic_df['macd'] = general.macd(tic_df['close'])
                elif fact == "kdj":
                    tic_df['kdj'] = general.kdj(tic_df, "KDJ_K")  # KDJ_D   KDJ_J
                elif fact == "wr":
                    tic_df['wr'] = general.wr(tic_df)
                elif fact == "psy":
                    tic_df['psy'] = general.psy(tic_df['close'], "PSY")
                elif fact == "atr":
                    tic_df['atr'] = general.atr(tic_df)
                elif fact == "bbi":
                    tic_df['bbi'] = general.bbi(tic_df['close'])
                elif fact == "dmi":
                    tic_df['dmi'] = general.dmi(tic_df, "DMI_PDI")
                elif fact == "taq":
                    tic_df['taq'] = general.taq(tic_df, "TAQ_MID")
                elif fact == "ktn":
                    tic_df['ktn'] = general.ktn(tic_df, "KTN_mid")
                elif fact == "trix":
                    tic_df['trix'] = general.trix(tic_df['close'], "TRMA")
                elif fact == "vr":
                    tic_df['vr'] = general.vr(tic_df)
                elif fact == "emv":
                    tic_df['emv'] = general.emv(tic_df, "MAEMV")
                elif fact == "dpo":
                    tic_df['dpo'] = general.dpo(tic_df['close'], "DPO")
                elif fact == "brar":
                    tic_df['brar'] = general.brar(tic_df)
                elif fact == "dfma":
                    tic_df['dfma'] = general.dfma(tic_df['close'])
                elif fact == "mtm":
                    tic_df['mtm'] = general.mtm(tic_df['close'], "MTM")
                elif fact == "mass":
                    tic_df['mass'] = general.mass(tic_df, "MASS")
                elif fact == "obv":
                    tic_df['obv'] = general.obv(tic_df)
                elif fact == "mfi":
                    tic_df['mfi'] = general.mfi(tic_df)
                elif fact == "asi":
                    tic_df['asi'] = general.asi(tic_df, "ASI")
                elif fact == "xsii":
                    tic_df['xsii'] = general.xsii(tic_df, "XSII_TD1")

            final_df = final_df.append(tic_df)

        return final_df

    def add_turbulence(self):
        self.processor.add_turbulence()
        self.dataframe = self.processor.dataframe

    def add_vix(self):
        self.processor.add_vix()
        self.dataframe = self.processor.dataframe

    def df_to_array(self, if_vix: bool) -> np.array:
        price_array, tech_array, turbulence_array = self.processor.df_to_array(
            self.tech_indicator_list, if_vix
        )
        # fill nan with 0 for technical indicators
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        return price_array, tech_array, turbulence_array

    def data_split(self, df, start, end, target_date_col="date"):
        """
        split the dataset into training or testing using date
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
        """
        data = df[(df[target_date_col] >= start) & (df[target_date_col] <= end)]
        data = data.sort_values([target_date_col, "tic"], ignore_index=True)
        data.index = data[target_date_col].factorize()[0]
        return data

    def run(
            self,
            ticker_list: str,
            technical_indicator_list: List[str],
            if_vix: bool,
            cache: bool = False,
            # file_dir: str = "./cache",
            # file_name: str = "",
            select_stockstats_talib: int = 0,
    ):
        cache_filename = (
                "_".join(
                    ticker_list
                    + [
                        self.data_source,
                        self.start_date,
                        self.end_date,
                        self.time_interval,
                    ]
                )
                + ".pickle"
        )
        cache_dir = "./cache"
        cache_path = os.path.join(cache_dir, cache_filename)

        if self.time_interval == "1s" and self.data_source != "binance":
            raise ValueError(
                "Currently 1s interval data is only supported with 'binance' as data source"
            )

        if cache and os.path.isfile(cache_path):
            print(f"Using cached file {cache_path}")
            self.tech_indicator_list = technical_indicator_list
            with open(cache_path, "rb") as handle:
                self.processor.dataframe = pickle.load(handle)
        else:
            self.download_data(ticker_list)
            self.clean_data()
            print(self.dataframe)
            if cache:
                if not os.path.exists(cache_dir):
                    os.mkdir(cache_dir)
                with open(cache_path, "wb") as handle:
                    pickle.dump(
                        self.dataframe,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

                self.dataframe.to_csv(cache_path + ".csv", index=False)

        self.add_technical_indicator(technical_indicator_list, select_stockstats_talib)
        if if_vix:
            self.add_vix()
        price_array, tech_array, turbulence_array = self.df_to_array(if_vix)
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        print(self.dataframe)
        return price_array, tech_array, turbulence_array

    def run_download(
            self,
            ticker_list: str,
            cache: bool = False,
            file_dir: str = "./cache",
            file_name: str = "",

    ):
        file_path = os.path.join(file_dir, file_name)
        if self.time_interval == "1s" and self.data_source != "binance":
            raise ValueError(
                "Currently 1s interval data is only supported with 'binance' as data source"
            )

        # cache_filename = (
        #         "_".join(
        #             ticker_list
        #             + [
        #                 self.data_source,
        #                 self.start_date,
        #                 self.end_date,
        #                 self.time_interval,
        #             ]
        #         )
        #         + ".pickle"
        # )
        # cache_dir = "./cache"
        # cache_path = os.path.join(cache_dir, cache_filename)

        if cache and os.path.isfile(file_path):
            print(f"Using cached file {file_path}")
            # self.tech_indicator_list = technical_indicator_list
            with open(file_path, "rb") as handle:
                self.processor.dataframe = pickle.load(handle)
                self.dataframe = self.processor.dataframe
        else:
            self.download_data(ticker_list)
            self.clean_data()
            # print(self.dataframe)
            if cache:
                if not os.path.exists(file_dir):
                    os.mkdir(file_dir)
                with open(file_path, "wb") as handle:
                    pickle.dump(
                        self.dataframe,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                self.dataframe.to_csv(file_path + ".csv", index=False)

        print(self.dataframe)

    def run_fileload(
            self,
            if_vix: bool = False,
            file_dir: str = "./cache",
            file_name: str = "",
    ):
        cache_path = os.path.join(file_dir, file_name)
        if os.path.isfile(cache_path):
            # print(f"Using cached file {cache_path}")
            # self.processor.dataframe = pd.read_csv(cache_path, parse_dates=['date'])
            with open(cache_path, "rb") as handle:
                self.processor.dataframe = pickle.load(handle)
                self.dataframe = self.processor.dataframe

        # self.add_technical_indicator(technical_indicator_list, select_stockstats_talib)
        if if_vix:
            self.add_vix()

    def data_load(
            self,
            ticker_list: str,
            technical_indicator_list: List[str],
            if_vix: bool = False,
            file_dir: str = "./cache",
            file_name: str = "",
            select_stockstats_talib: int = 0,
    ):
        self.tech_indicator_list = technical_indicator_list
        load_file_dir = file_dir
        load_file_name = ""
        if file_name != "":
            load_file_name = file_name
        else:
            tmp_filename = (
                    "_".join(
                        ticker_list
                        + [
                            self.data_source,
                            self.start_date,
                            self.end_date,
                            self.time_interval,
                        ]
                    )
                    + ".csv"
            )
            load_file_name = tmp_filename

        cache_path = os.path.join(load_file_dir, load_file_name)
        if os.path.isfile(cache_path):
            print(f"Using cached file {cache_path}")
            self.processor.dataframe = pd.read_csv(cache_path, parse_dates=['date'])
            # with open(cache_path, "rb") as handle:
            # self.processor.dataframe = pickle.load(handle)

        self.add_technical_indicator(technical_indicator_list, select_stockstats_talib)
        if if_vix:
            self.add_vix()
        price_array, tech_array, turbulence_array = self.df_to_array(if_vix)
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        print(self.dataframe)
        return price_array, tech_array, turbulence_array


def test_joinquant():
    # TRADE_START_DATE = "2019-09-01"
    TRADE_START_DATE = "2020-09-01"
    TRADE_END_DATE = "2021-09-11"

    # supported time interval: '1m', '5m', '15m', '30m', '60m', '120m', '1d', '1w', '1M'
    TIME_INTERVAL = "1d"
    TECHNICAL_INDICATOR = [
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
    ]

    kwargs = {"username": "xxx", "password": "xxx"}
    p = DataProcessor(
        data_source="joinquant",
        start_date=TRADE_START_DATE,
        end_date=TRADE_END_DATE,
        time_interval=TIME_INTERVAL,
        **kwargs,
    )

    ticker_list = ["000612.XSHE", "601808.XSHG"]

    p.download_data(ticker_list=ticker_list)

    p.clean_data()
    p.add_turbulence()
    p.add_technical_indicator(TECHNICAL_INDICATOR)
    p.add_vix()

    price_array, tech_array, turbulence_array = p.run(
        ticker_list, TECHNICAL_INDICATOR, if_vix=False, cache=True
    )
    pass

# if __name__ == "__main__":
#     # test_joinquant()
#     # test_binance()
#     # test_yahoofinance()
#     test_baostock()
#     # test_quandl()
