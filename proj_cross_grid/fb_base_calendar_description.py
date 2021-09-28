import numpy as np
import pandas as pd
import os
DATA_ROOT = "/share/xfzhang/store_data/"

__all__ = ['trade_calendar', 'stk_description', 'idx_description', 'df_sid_change']


def load_calandar_descr():
    path = os.path.join(DATA_ROOT, 'base_data/calendar')
    with open(path, 'r') as f:
        dates = [line.strip() for line in f]
    arr_calendar = np.array(dates).astype('i4')
            
    path = os.path.join(DATA_ROOT, 'base_data/stock_description')
    with open(path, 'r') as f:
        stk_descr = [line.strip().split(' ') for line in f]
    df_stk_dscr = pd.DataFrame(stk_descr, columns=['sid', 'list_date', 'delist_date'])
    df_stk_dscr['list_date'] = df_stk_dscr['list_date'].values.astype('i4')
    df_stk_dscr['delist_date'] = df_stk_dscr['delist_date'].values.astype('i4')
            
    path = os.path.join(DATA_ROOT, 'base_data/index_description')
    with open(path, 'r') as f:
        idx_descr = [line.strip().split(' ') for line in f]
    df_idx_dscr = pd.DataFrame(idx_descr, columns=['iid', 'list_date', 'expire_date'])
    df_idx_dscr['list_date'] = df_idx_dscr['list_date'].values.astype('i4')
    df_idx_dscr['expire_date'] = df_idx_dscr['expire_date'].values.astype('i4')
    
    path = os.path.join(DATA_ROOT, 'base_data/sid_change')
    with open(path, 'r') as f:
        sid_change = [line.strip().split(' ') for line in f]
    df_sid_change = pd.DataFrame(sid_change, columns=['date', 'old_sid', 'new_sid'])
    df_sid_change['date'] = df_sid_change['date'].values.astype('i4')
            
    return arr_calendar, df_stk_dscr, df_idx_dscr, df_sid_change

arr_calendar, df_stk_dscr, df_idx_dscr, df_sid_change = load_calandar_descr()


class TradeCalendar:
    def __init__(self):
        self.trade_dates = arr_calendar

    def get_prev_trade_date(self, date: int, n=1):
        pos = self.trade_dates.searchsorted(int(date))
        if pos < n:
            raise MemoryError
        return self.trade_dates[pos-n]

    def get_next_trade_date(self, date: int):
        pos = self.trade_dates.searchsorted(int(date), side='right')
        return self.trade_dates[pos]

    def get_trade_dates(self, start_date, end_date):
        s = np.searchsorted(self.trade_dates, int(start_date))
        e = np.searchsorted(self.trade_dates, int(end_date), side='right')
        return self.trade_dates[s:e].copy()
    

class StockDescription():
    def __init__(self):
        self.data = df_stk_dscr
        
    def get_dates_sids(self, start_date, end_date=None, n=0):  #list date>=n, not include start_date
        end_date = start_date if end_date is None else end_date
        start_date, end_date = int(start_date), int(end_date)
        s_pos = np.searchsorted(trade_calendar.trade_dates, start_date)
        e_pos = np.searchsorted(trade_calendar.trade_dates, end_date, side='right') - 1
        assert s_pos >= n and s_pos <= e_pos
        start_date0 = trade_calendar.trade_dates[s_pos - n]
        end_date0 = trade_calendar.trade_dates[e_pos - n]
        
        df_raw = self.data
        unique_dates = trade_calendar.get_trade_dates(start_date0, end_date)
        df = df_raw[(df_raw.list_date <= end_date0) & (df_raw.delist_date > start_date)]
        unique_sids = df.sid.values.astype('U9')
        
        arr = np.zeros([len(unique_dates), len(unique_sids)], dtype='i4')
        pos = np.searchsorted(unique_sids, df.sid)
        pos_up = np.searchsorted(unique_dates, df.list_date)
        pos_dn = np.searchsorted(unique_dates, df.delist_date)
        for i in range(len(df)):
            #if pos_up[i] + n < len(arr):
            arr[pos_up[i] + n, pos[i]] += 1
            if pos_dn[i] < len(arr):
                arr[pos_dn[i], pos[i]] -= 1
        
        arr_cum = np.cumsum(arr, axis=0)
        dates_pos, sids_pos = np.where(arr_cum[n:]>=1)
        # dates = unique_dates[dates_pos]
        # sids = unique_sids[sids_pos]
        return unique_dates[n:], unique_sids, dates_pos, sids_pos


class IndexDescription:
    def __init__(self):
        self.data = df_idx_dscr

    def get_dates_iids(self, start_date, end_date=None):
        end_date = start_date if end_date is None else end_date
        start_date, end_date = int(start_date), int(end_date)
        df_raw = self.data
        unique_dates = trade_calendar.get_trade_dates(start_date, end_date)
        df = df_raw[(df_raw.list_date <= end_date) & (df_raw.expire_date > start_date)]
        unique_iids = df.iid.values.astype('U9')
        
        arr = np.zeros([len(unique_dates), len(unique_iids)], dtype='i4')
        pos = np.searchsorted(unique_iids, df.iid)
        pos_up = np.searchsorted(unique_dates, df.list_date)
        pos_dn = np.searchsorted(unique_dates, df.expire_date)
        for i in range(len(df)):
            if pos_up[i] < len(arr):
                arr[pos_up[i], pos[i]] += 1
            if pos_dn[i] < len(arr):
                arr[pos_dn[i], pos[i]] -= 1
        
        arr_cum = np.cumsum(arr, axis=0)
        dates_pos, iids_pos = np.where(arr_cum>=1)
        # dates = unique_dates[dates_pos]
        # iids = unique_iids[iids_pos]
        return unique_dates, unique_iids, dates_pos, iids_pos  

    
trade_calendar = TradeCalendar()
stk_description = StockDescription()
idx_description = IndexDescription()

if __name__ == '__main__':
    _, sids = stk_description.get_sids('20181228')
    _, iids = idx_description.get_iids('20181228')
    print(len(sids), len(iids))