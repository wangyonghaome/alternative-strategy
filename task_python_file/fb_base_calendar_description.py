import numpy as np
import pandas as pd
import os
DATA_ROOT = "/share/xfzhang/store_data/"

__all__ = ['trade_calendar', 'stk_description', 'idx_description', 'sid_change', 'df_sid_change']


class TradeCalendar:
    def __init__(self):
        path = os.path.join(DATA_ROOT, 'base_data/calendar')
        with open(path, 'r') as f:
            dates = [line.strip() for line in f]
        self.trade_dates = np.array(dates).astype('i4')

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
        path = os.path.join(DATA_ROOT, 'base_data/stock_description')
        df_stk_dscr = pd.read_csv(path)
        df_stk_dscr['list_date'] = df_stk_dscr['list_date'].values.astype('i4')
        df_stk_dscr['delist_date'] = df_stk_dscr['delist_date'].values.astype('i4')
        self.data = df_stk_dscr
        self.map_sid_name = {row['sid']: row['name'] for _, row in df_stk_dscr.iterrows()}
        
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
        return unique_dates[n:], unique_sids, dates_pos, sids_pos


class IndexDescription:
    def __init__(self):
        path = os.path.join(DATA_ROOT, 'base_data/index_description')
        df_idx_dscr = pd.read_csv(path)
        df_idx_dscr['list_date'] = df_idx_dscr['list_date'].values.astype('i4')
        df_idx_dscr['expire_date'] = df_idx_dscr['expire_date'].values.astype('i4')
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
        return unique_dates, unique_iids, dates_pos, iids_pos  
    
class SidChange:
    def __init__(self):
        path = os.path.join(DATA_ROOT, 'base_data/sid_change')
        df_sid_change = pd.read_csv(path)
        df_sid_change['date'] = df_sid_change['date'].values.astype('i4')
        self.data = df_sid_change
        self.map_sid_change = {row['old_sid']: row['new_sid'] for _, row in df_sid_change.iterrows()}

    
trade_calendar = TradeCalendar()
stk_description = StockDescription()
idx_description = IndexDescription()
sid_change = SidChange()
df_sid_change = sid_change.data.copy()


if __name__ == '__main__':
    _, sids = stk_description.get_sids('20181228')
    _, iids = idx_description.get_iids('20181228')
    print(len(sids), len(iids))