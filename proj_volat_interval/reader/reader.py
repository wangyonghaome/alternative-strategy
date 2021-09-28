from __future__ import division

import datetime
import numpy as np
import yaml

from .common import sessions, bartype


def generate_labels(freq, sessions):
    date = datetime.date.today()
    d = datetime.timedelta(minutes=freq)
    t = []
    for start, end in sessions:
        s = datetime.datetime.combine(date, start) + d
        e = datetime.datetime.combine(date, end) + d
        a = np.arange(np.datetime64(s), np.datetime64(e), np.timedelta64(d)).astype(datetime.datetime)
        t.append(a)

    labels = np.array([int(e.strftime('%H%M%S')) * 1000 for e in np.concatenate(t)])
    return labels


class BarReader:

    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.stock_bar_store = None
        self.index_bar_store = None

        self.stock_sids = {}
        self.index_sids = {}

        self.stock_symbols = []
        self.delisted_symbols = set()
        self.listed_symbols = set()

    def __enter__(self):
        with open(self.config_file_path,mode = "r") as f:
            config = yaml.load(f, Loader=yaml.Loader)

        # symbol maps
        symbol_config = config['symbol']
        with open(symbol_config['stock'],mode = "r") as f:
            for i, line in enumerate(f):
                symbol = line[:6]
                self.stock_symbols.append(symbol)
                self.stock_sids[symbol] = i

        with open(symbol_config['index'],mode = "r") as f:
            for i, line in enumerate(f):
                symbol = line[:6]
                self.index_sids[symbol] = i

        with open(symbol_config['delisted'],mode = "r") as f:
            for i, line in enumerate(f):
                symbol = line.strip()
                self.delisted_symbols.add(symbol)

        for symbol in self.stock_symbols:
            if symbol not in self.delisted_symbols:
                self.listed_symbols.add(symbol)

        # stock tick store
        stock_config = config['generator']['stock']
        self.n_stock_bars = 240 // stock_config['freq']
        self.stock_bar_store = np.memmap(
            stock_config['store_file'],
            mode='r',
            dtype=bartype,
            shape=(stock_config['size'] * self.n_stock_bars,)
        )

        self.stock_bar_lock = np.memmap(
            stock_config['lock_file'],
            mode='r',
            dtype=np.uint8,
            shape=(stock_config['size'],)
        )

        self.stock_labels = generate_labels(stock_config['freq'], sessions)

        # index tick store
        index_config = config['generator']['index']
        self.n_index_bars = 240 // stock_config['freq']
        self.index_bar_store = np.memmap(
            index_config['store_file'],
            mode='r',
            dtype=bartype,
            shape=(index_config['size'] * self.n_index_bars,)
        )

        self.index_bar_lock = np.memmap(
            stock_config['lock_file'],
            mode='r',
            dtype=np.uint8,
            shape=(index_config['size'],)
        )

        self.index_labels = generate_labels(index_config['freq'], sessions)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def stock_bars(self, symbol):
        n = self.n_stock_bars
        sid = self.stock_sids[symbol]

        lock = self.stock_bar_lock
        start = sid * n
        end = start + n

        while True:
            while True:
                seq = lock[sid]
                if seq & 1:
                    continue
                break
                
            bars = self.stock_bar_store[start:end].copy()
            if lock[sid] == seq:
                break

        bars['preclose'][1:n] = bars['close'][0: n - 1]
        bars['time'] = self.stock_labels
        return bars

    def index_bars(self, symbol):
        n = self.n_index_bars
        sid = self.index_sids[symbol]

        lock = self.index_bar_lock
        start = sid * n
        end = start + n

        while True:
            while True:
                seq = lock[sid]
                if seq & 1:
                    continue
                break

            bars = self.index_bar_store[start:end].copy()
            if lock[sid] == seq:
                break

        bars['preclose'][1:n] = bars['close'][0:n-1]
        bars['time'] = self.index_labels
        return bars

    def listed_symbols(self):
        return self.listed_symbols
