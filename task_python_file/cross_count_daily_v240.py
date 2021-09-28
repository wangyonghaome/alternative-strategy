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
from fb_base_get_market_data import *
from fb_base_calendar_description import *
from data_tools.api import *
from utilscht.Data import *
import pymysql
import datetime
import logging
import click
from reader.reader import BarReader

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
    if np.isnan(total_seq).sum()>10:
        return np.nan
    
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
        DTYPE_STK_BAR_1MIN = [
            ('wcode', 'S16'),
            ('date', 'i4'),
            ('time', 'i4'),
            ('open', 'i4'),
            ('high', 'i4'),
            ('low', 'i4'),
            ('close', 'i4'),
            ('volume', 'i8'),
            ('value', 'i8'),
            ('num_trades', 'i4'),
            ('unused', 'i4'),
        ]



        bar=np.memmap(r"/dat/raw/wind/stk_bar_1min/{}/{}.dat".format(date,sid),dtype=DTYPE_STK_BAR_1MIN,mode="r")
        df = pd.DataFrame(bar)[["wcode","date","time","close"]]
        df["sid"] = df["wcode"].apply(lambda x:str(x)[2:-1])
        df["DataDate"] = df["date"].apply(str)
        df["close"] = df["close"]/10000
        df["pre_close"] = get_stk_bar(sid, freq="1d", start=date, end=date, fields=["pre_close"]).values[0][0]
        df = df.drop(["wcode","date","time"],axis=1)
        df["ticktime"] = df.index
        return df[["sid","DataDate","ticktime","close","pre_close"]]
    except Exception as e:
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
    df=get_db(start_date = date, end_date = date,cols=["trade_status"])
    stk_ls=list(df["sid"][df["trade_status"]==1])

    results = Parallel(n_jobs=16, verbose=5, backend="loky", batch_size='auto') \
        (delayed(get_stk_data)(sid, date) for sid in stk_ls)
    data = pd.concat(results)

    return data.sort_values(["sid", "ticktime"])

def get_today_stk_data(sid,date):
    try:
        with BarReader('/home/ywang/proj_cross_grid/reader/bar.yaml') as client:
            bars = client.stock_bars(sid[0:6])
            df = pd.DataFrame(bars)
            df["sid"] = df["symbol"].apply(lambda x:str(x)[2:-1])
            df["DataDate"] = df["date"].apply(str)
            df["ticktime"] = df.index
            df = df.drop(["symbol","date","time"],axis=1)
            df["pre_close"] = df["preclose"].iloc[0]
            df= df[df["sid"] != ""]
            return df[["sid","DataDate","ticktime","close","pre_close"]]
    except:
        logging.warning("Data Lost for {} on {}".format(sid,date))


def get_today_data(date):
    pre_date = get_previous_trade_date(date)
    df=get_db(start_date = pre_date, end_date = pre_date,cols=["trade_status"])
    stk_ls=list(df["sid"][df["trade_status"] == 1])

    results = Parallel(n_jobs=16, verbose=5, backend="loky", batch_size='auto') \
        (delayed(get_today_stk_data)(sid,date) for sid in stk_ls)
    data = pd.concat(results)

    return data.sort_values(["sid", "ticktime"])


def get_count_parallel(df):
    if sum(pd.isnull(df["yesterday_close"]))>120:
        return pd.DataFrame()
    sid = df["sid"].drop_duplicates()
    temp_df = pd.DataFrame(index=sid, columns=["cross_count"])
    
    pre_close = df["pre_close"].iloc[0]
    if pre_close!=df["yesterday_close"].iloc[-1]:
        df["yesterday_close"] = df["yesterday_close"]* pre_close/df["yesterday_close"].iloc[-1]

    close_seq = np.concatenate([np.array(df["yesterday_close"].iloc[-40:]),\
                                np.array(df["close"].iloc[:200])])
    close_seq = close_seq[~np.isnan(close_seq)]
    pre_close_seq = np.array(df["yesterday_close"].iloc[-240:-40])
    
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

    data = get_today_data(date)
    data["sid"] = data["sid"].apply(lambda x:str(x)+".SH" if str(x)[0]=='6' else str(x)+'.SZ')
    data["DataDate"] = data["DataDate"].apply(str)
    yesterday_data = get_daily_data(pre_date)
    yesterday_data.rename(columns={"close": "yesterday_close"}, inplace=True)
    data = pd.merge(data, yesterday_data[["sid", "ticktime", "yesterday_close"]], how="left", on=["sid", "ticktime"])

    groups = data.groupby("sid")
    cross_count_df = apply_parallel(groups, get_count_parallel)
    cross_count_df.to_csv(r"/home/ywang/proj_cross_grid/result/cross_count_b240/crosscount_{}.csv".format(date), index_label="sid")    
    
if __name__ == '__main__':
    get_daily_cross_data()
