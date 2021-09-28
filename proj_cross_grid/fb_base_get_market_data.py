import numpy as np
import pandas as pd
import os
import bcolz

DATA_ROOT = "/share/xfzhang/store_data/"


col_dtype_dflt = [('date', 'i4', 0), ('sid', 'S9', b'000000000'),  ('time', 'i4', 0),
                  
                  ('pre_close', 'f8', np.nan), ('open', 'f8', np.nan), ('high', 'f8', np.nan), 
                  ('low', 'f8', np.nan), ('close', 'f8', np.nan), ('amount', 'f8', np.nan), ('vwap', 'f8', np.nan),
                  
                  ('amount_mean', 'f8', 0.0), ('amount_std', 'f8', 0.0),  # ('trans_num', 'i8', 0), 
                  ('ask_id_amount_mean', 'f8', 0.0), ('ask_id_amount_std', 'f8', 0.0),  # ('ask_id_amount_mean', 'f8', 0.0), ('bid_id_trans_num', 'i8', 0), 
                  ('bid_id_amount_mean', 'f8', 0.0), ('bid_id_amount_std', 'f8', 0.0),
                  
                  ('act_ask_amount_sum', 'i8', 0),  ('act_ask_amount_mean', 'f8', 0.0), ('act_ask_amount_std', 'f8', 0.0),
                  ('act_bid_amount_sum', 'i8', 0),  ('act_bid_amount_mean', 'f8', 0.0), ('act_bid_amount_std', 'f8', 0.0),
                  ('act_ask_id_amount_mean', 'f8', 0.0), ('act_ask_id_amount_std', 'f8', 0.0),
                  ('act_bid_id_amount_mean', 'f8', 0.0), ('act_bid_id_amount_std', 'f8', 0.0),
                  
                  ('tot_ask_amount1', 'f8', np.nan), ('avg_ask_price1', 'f8', np.nan),
                  ('tot_bid_amount1', 'f8', np.nan), ('avg_bid_price1', 'f8', np.nan),
                  ('tot_ask_amount2', 'f8', np.nan), ('avg_ask_price2', 'f8', np.nan),
                  ('tot_bid_amount2', 'f8', np.nan), ('avg_bid_price2', 'f8', np.nan),
                  ('tot_ask_amount5', 'f8', np.nan), ('avg_ask_price5', 'f8', np.nan),
                  ('tot_bid_amount5', 'f8', np.nan), ('avg_bid_price5', 'f8', np.nan),
                  ('tot_ask_amount10', 'f8', np.nan), ('avg_ask_price10', 'f8', np.nan),
                  ('tot_bid_amount10', 'f8', np.nan), ('avg_bid_price10', 'f8', np.nan),
                  ('tot_ask_amount', 'f8', np.nan), ('avg_ask_price', 'f8', np.nan),
                  ('tot_bid_amount', 'f8', np.nan), ('avg_bid_price', 'f8', np.nan),
                  
                  ('close_suspend', 'i4', 0), ('close_limit', 'i4', 0), ('all_limit', 'i4', 0),
                 ]

col_dtype_dflt_idx = [('date', 'i4', 0), ('iid', 'S9', b'000000000'),  ('time', 'i4', 0),
                  
                  ('pre_close', 'f8', np.nan), ('open', 'f8', np.nan), ('high', 'f8', np.nan), 
                  ('low', 'f8', np.nan), ('close', 'f8', np.nan), ('volume', 'f8', np.nan), ('amount', 'f8', np.nan),
                 ]


def get_db(start_date, end_date, cols=None): 
    path = f'{DATA_ROOT}/bcolz_data/1d/stock_data'
    table = bcolz.open(path, mode='r')
    dates = table['date']
    columns = [ele for ele in table.names if ele != '_id']  # bcolz
    s = np.searchsorted(dates, int(start_date))
    e = np.searchsorted(dates, int(end_date), side='right')
    if cols is not None:
        assert len(set(cols).difference(columns)) == 0
        cols = [ele for ele in columns if ele in ['sid', 'date'] + cols]
    else:
        cols = columns
    df = pd.DataFrame(table[cols][s:e])
    df.sid = df.sid.values.astype('U9')
    return df


def get_idx_db(start_date, end_date, cols=None): 
    path = f'{DATA_ROOT}/bcolz_data/1d/index_data'
    table = bcolz.open(path, mode='r')
    dates = table['date']
    columns = [ele for ele in table.names if ele != '_id']  # bcolz
    s = np.searchsorted(dates, int(start_date))
    e = np.searchsorted(dates, int(end_date), side='right')
    if cols is not None:
        assert len(set(cols).difference(columns)) == 0
        cols = [ele for ele in columns if ele in ['iid', 'date'] + cols]
    else:
        cols = columns
    df = pd.DataFrame(table[cols][s:e])
    df.iid = df.iid.values.astype('U9')
    return df


def get_mb(start_date, end_date,  freq='15m', fields=None): 
    trade_dates = trade_calendar.get_trade_dates(start_date, end_date)
    path = f"{DATA_ROOT}/np_data/{freq}/stock_data"
    dtype = [(ele[0], ele[1]) for ele in col_dtype_dflt]
    arr_all = []
    if fields is None:
        fields = [ele[0] for ele in dtype]
    fields2 = [ele[0] for ele in dtype if ele[0] in fields+['date', 'sid', 'time']]
    for i, date in enumerate(trade_dates):
        year = date // 10000
        path2 = f"{path}/{year}/{date}.dat"
        arr = np.memmap(path2, dtype=dtype, mode='r')
        arr_all.append(arr[fields2])
    arr_all = np.concatenate(arr_all)
    return arr_all

def get_idx_mb(start_date, end_date,  freq='15m', fields=None): 
    trade_dates = trade_calendar.get_trade_dates(start_date, end_date)
    path = f"{DATA_ROOT}/np_data/{freq}/index_data"
    dtype = [(ele[0], ele[1]) for ele in col_dtype_dflt_idx]
    arr_all = []
    if fields is None:
        fields = [ele[0] for ele in dtype]
    fields2 = [ele[0] for ele in dtype if ele[0] in fields+['date', 'iid', 'time']]
    for i, date in enumerate(trade_dates):
        year = date // 10000
        path2 = f"{path}/{year}/{date}.dat"
        arr = np.memmap(path2, dtype=dtype, mode='r')
        arr_all.append(arr[fields2])
    arr_all = np.concatenate(arr_all)
    return arr_all