#!/home/ywang/env/dev/bin
#-*-coding:utf-8-*-
# written by wangyonghao

# the api: get_daily_cross_data(date)

# input:  date: current date ( "20190102" for example)

# output to file: the csv file of the cross_grid_count for each stock

import numpy as np
import pandas as pd
import os
from joblib import Parallel, delayed
import tushare as ts
from data_tools.api import *
from utilscht.Data import *
import pymysql
import click
import logging
import datetime

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename=r'logging/cross_count_daily.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'#日志格式
                   )

def get_cross_grid_info(price_now, ref_price,grids):
    if price_now > ref_price:
        cross_grids = [i for i in grids if i > ref_price and i <= price_now]
        if len(cross_grids) > 0:
            grids_num = len(cross_grids)
            direction = 1
            last_grid = cross_grids[-1]

    else:
        cross_grids = [i for i in grids if i > price_now and i <= ref_price]
        if len(cross_grids) > 0:
            grids_num = len(cross_grids)
            direction = -1
            last_grid = cross_grids[0]

    if len(cross_grids) > 0:
        return (grids_num, direction, last_grid)
    else:
        return -1


# 输入前一天最后 60min 的close序列、今天的全部close序列、前一天的收盘价、今天的开盘价
# ，返回今天穿越网格线的次数（一段时间内重复穿过同一条线只计算一次）
def get_cross_count(pre_close_seq, close_seq):
    total_seq=np.concatenate([pre_close_seq,close_seq])
    if (np.nanmin(total_seq)+np.nanmax(total_seq))/2>=10:
        grid_unit=(np.nanmin(total_seq)+np.nanmax(total_seq))/2*0.01
    else:
        grid_unit = 0.1
    grid_num = int((np.nanmax(total_seq)-np.nanmin(total_seq))/grid_unit+3)
    grids = np.linspace(np.nanmin(total_seq)-grid_unit,np.nanmax(total_seq)+grid_unit,grid_num)
    
    count = 0
    ref_price = 0
    pre_cross_dire = 1

    # get the last corss_line for yesterday
    for i in reversed(range(len(pre_close_seq))):
        price_pre = pre_close_seq[i - 1]
        price_now = pre_close_seq[i]
        cross_grid_info = get_cross_grid_info(price_now, price_pre,grids)

        if cross_grid_info != -1:
            ref_price = cross_grid_info[2]
            pre_cross_dire = cross_grid_info[1]
            break

    # count the cross times for today
    close_seq = np.concatenate([np.array([pre_close_seq[-1]]),close_seq])
    for i in range(len(close_seq) - 1):
        price_pre = close_seq[i]
        price_now = close_seq[i + 1]

        cross_grid_info = get_cross_grid_info(price_now, price_pre,grids)

        if cross_grid_info == -1:
            continue

        # judge the status of "now" and "pre"
        grids_num = cross_grid_info[0]
        cross_dire = cross_grid_info[1]
        last_grid = cross_grid_info[2]

        if last_grid != ref_price:
            if cross_dire * pre_cross_dire == 1:
                count = count + grids_num
            else:
                count = count + grids_num - 1

        ref_price = last_grid
        pre_cross_dire = cross_dire

    return count


# current function to get 1min_freq data
def get_stk_data(sid, date):
    
    try:
        raw_bar_dtype = [
            ('sid', '<U16'),
            ('DataDate', 'i4'),
            ('ticktime', 'i4'),
            ('pre_close', 'f8'),
            ('open', 'f8'),
            ('high', 'f8'),
            ('low', 'f8'),
            ('close', 'f8'),
            ('volume', 'f8'),
            ('amount', 'f8'),
            ('bid_amount', 'f8'),
            ('ask_amount', 'f8'),
            ('vwap', 'f8'),
            ('twap', 'f8'),
            ('ret', 'f8')
        ]

        bar=np.memmap(r"/share/intern_share/dat/stk_bar/1min/2019/{}/{}.dat".\
                  format(date,sid),dtype=raw_bar_dtype,mode="r")
        stock_data=pd.DataFrame(bar)[["sid","DataDate","ticktime","open", "high", "low", "close","pre_close"]]
        stock_data["ticktime"]=stock_data.index
        return stock_data
    except:
        try:
            stock_data = get_stk_bar(sid, freq="1m", start=date, end=date, fields=["open", "high", "low", "close"])
            stock_data = stock_data.reset_index().rename(columns={"index": "datetime"})
            stock_data["ticktime"] = stock_data["datetime"].index
            stock_data["DataDate"] = stock_data["datetime"].apply(lambda x:str(x.date()))
            del stock_data["datetime"]
            stock_data["pre_close"] = get_stk_bar(sid, freq="1d", start=date, end=date, fields=["pre_close"]).values[0][0]
            stock_data["sid"] = sid
            return stock_data
        except:
            logging.warning("Data Lost for {} on {}".format(sid,date))


def get_daily_data(date):
    df=query_table("DailyBar",start_date=date,end_date=date,fields=["tradable"])
    stk_ls=list(df["sid"][df["tradable"]==1])

    results = Parallel(n_jobs=16, verbose=5, backend="loky", batch_size='auto') \
        (delayed(get_stk_data)(sid, date) for sid in stk_ls)
    data = pd.concat(results)

    return data.sort_values(["sid", "ticktime"])


def get_count_parallel(df):
    sid = df["sid"].drop_duplicates()
    temp_df = pd.DataFrame(index=sid, columns=["cross_count"])

    close_seq = np.array(df["close"])
    pre_close_seq = np.array(df["yesterday_close"].iloc[-60:])
    pre_close = df["pre_close"].iloc[0]
    if pre_close!=pre_close_seq[-1]:
        pre_close_seq = pre_close_seq* pre_close/pre_close_seq[-1]

    count = get_cross_count(pre_close_seq, close_seq)

    temp_df.iloc[0, 0] = count
    return temp_df


def apply_parallel(df_grouped, func, n_jobs=16, backend='loky', as_index=False, **kwargs):
    """
    This is for parallel between grouped generated by pd.DataFrame.groupby
    :param df_grouped:
    :param func:
    :param n_jobs:
    :param backend:
    :param kwargs:
    :return:
    """

    names = []
    groups = []
    for name, group in df_grouped:
        names.append(name)
        groups.append(group)

    results = Parallel(n_jobs=n_jobs, verbose=5, backend=backend, batch_size='auto') \
        (delayed(func)(group, **kwargs) for group in groups)

    return pd.concat(results, keys=names if as_index else None)



@click.command()
@click.argument("date",nargs=1)
def get_daily_cross_data(date):
    pre_date = get_previous_trade_date(date)

    data = get_daily_data(date)
    yesterday_data = get_daily_data(pre_date)
    yesterday_data.rename(columns={"close": "yesterday_close"}, inplace=True)
    data = pd.merge(data, yesterday_data[["sid", "ticktime", "yesterday_close"]], how="left", on=["sid", "ticktime"])

    groups = data.groupby("sid")
    cross_count_df = apply_parallel(groups, get_count_parallel)
    cross_count_df.to_csv(r"/share/intern_share/stk_bundle_v2/crosscount_{}.csv".format(date), index_label="sid")
    logging.info("{} finished".format(date))


if __name__ == '__main__':
    get_daily_cross_data()

