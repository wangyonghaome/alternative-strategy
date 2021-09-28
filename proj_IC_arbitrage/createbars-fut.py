# -*- coding: utf-8 -*-
# !/home/hcwu/anacoda3/bin/python

import os
import traceback

import ctpfields
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s :%(filename)s:%(lineno)s:%(levelname)s %(message)s')

default_instrument_file = "/home/hcwu/backup/instruments/20170316.csv"


def pydate(date):
    return datetime.datetime.strptime(str(date), "%Y%m%d").date()


def guess_exchange_from_ticker(ticker, date):
    if isinstance(ticker, (np.bytes_, bytes)):
        ticker = bytes(ticker).decode()
    global default_instrument_file
    import re
    self = guess_exchange_from_ticker
    #optpattern = r"(^.+-[CP]-.+$)|(^[a-zA-Z]\+\d+[a-zA-Z]\+\d+$)"
    if hasattr(self, 'exchmap'):
        exchmap = self.exchmap
    else:
        ctp_instruments_file = default_instrument_file
        date = pydate(date)
        n = 10
        while n > 0:
            fn = date.strftime("%Y%m%d") + ".csv"
            ctp_instruments_file = os.path.join(os.path.dirname(ctp_instruments_file), fn)
            if os.path.exists(ctp_instruments_file):
                break
            date -= datetime.timedelta(days=1)
            n -= 1

        if not os.path.exists(ctp_instruments_file):
            ctp_instruments_file = default_instrument_file

        if not os.path.exists(ctp_instruments_file):
            logging.error("ctp instrument file not found: %s", ctp_instruments_file)

        logging.info(f"using instrument file {ctp_instruments_file}")
        exchmap = pd.read_csv(ctp_instruments_file, encoding="gbk")[['InstrumentID', 'ExchangeID']]
        exchmap.InstrumentID = [re.sub(r"\d+$", "", x) if len(x) <= 6 else x  # ccYYMM == 6 chars
                                for x in exchmap.InstrumentID]
        exchmap = dict(zip(exchmap.InstrumentID, exchmap.ExchangeID))
        exchmap.update({
            'ME': 'CZCE',
            'TC': 'CZCE'
        })
        setattr(self, 'exchmap', exchmap)
    fs = re.sub(r"\d+$", "", ticker) if len(ticker) <= 6 else ticker
    return exchmap[fs]


def make_engine():
    MDSERVER_DB = ("192.168.1.190", 5432, "mdserver", "dataman", "123456")
    param = dict(list(zip("host,port,dbname,user,password".split(","), MDSERVER_DB)))
    import string
    engine = string.Template("postgresql://$user:$password@$host:$port/$dbname").substitute(param)
    return engine


def make_time(yyyymmdd, hhmmssppp):
    """

    :param yyyymmdd:
    :param hhmmssppp:
    :return:
     注意，不要在这儿改变 两个序列的长度，即不要做filter, 类似：
     yyyymmdd, hhmmssppp = yyyymmdd[ yyyymmdd > 19700101 ], hhmmssppp[yyyymmdd > 19700101 ]
    这会让调用者在赋于 df.index时报错，长度mismatch
    """
    import numpy as np
    import pandas as pd
    # yyyy = np.divide(yyyymmdd, 10000) # so yyyyy is int32,this is py2
    yyyy = yyyymmdd // 10000
    mm = np.mod(yyyymmdd, 10000) // 100
    dd = np.mod(yyyymmdd, 100)
    hh = hhmmssppp // int(1e7)
    MM = np.mod(hhmmssppp, int(1e7)) // int(1e5)
    ss = np.mod(hhmmssppp, int(1e5)) // int(1e3)
    ppp = np.mod(hhmmssppp, int(1e3)) * 1000  # millis => micros
    args = list(zip(yyyy, mm, dd, hh, MM, ss, ppp))
    index = [pd.datetime(*x) for x in args]
    return index


def read_buffer(hqxzroot, date, flat_fieldmap, skipbytes=0):
    """
    读取.hq or .hq.xz文件，返回 np.recarray对象
    :param hqxzroot:
    :param h5root:
    :param date:
    :param flat_fieldmap: one possible value is hhfields.stock.flat_fieldmap
    :param skipbytes: skip @skipbytes bytes, since old version has 32 bytes in header
    :return:
    """
    import os, subprocess
    f1 = os.path.join(hqxzroot, str(date) + ".hq.xz")
    f2 = os.path.join(hqxzroot, str(date) + ".hq")
    if os.path.exists(f1) and os.path.exists(f2):  # I am not sure if .xz is finished
        raise Exception("find both .hq.xz and .hq")
    if os.path.exists(f2):
        logging.info("read file %s", f2)
        fd = open(f2, "rb")
        # size =  hhfields.stock.flat_fieldmap.itemsize * 1024*1024
        data = fd.read()
        logging.info("file read done")
    elif os.path.exists(f1):
        args = ("""xz -kd --to-stdout %s """ % f1).split()
        pipe = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info("decompress file %s", f1)
        data, error = pipe.communicate()
        logging.info("decompress done")
        if error:
            raise Exception(error)
    else:
        logging.warning("none of the files exist! \nf1=%s\nf2=%s" % (f1, f2))
        return None
    logging.info("build np.records")
    if skipbytes > 0:
        data = data[skipbytes:]
    records = pd.np.frombuffer(data, dtype=flat_fieldmap)  # Stock special
    logging.info("np.records built")
    return records


class FutureFixup(object):
    def __init__(self, args):
        self.args = args

    def make_daily(self, hqxzroot, h5root, date):
        """
        从新版(由 HH C++程序接宏汇行情生成的.hq文件(或者压缩后的.hq.xz文件)的生成相应的日线数据。
        :param hqxzroot:
        :return:
        """
        cache = {}
        logging.info("build cache")
        import ctpfields
        flat_fieldmap = ctpfields.future.flat_fieldmap
        records = read_buffer(hqxzroot, h5root, date, flat_fieldmap)
        for r in records:
            key = r["ticker"]
            cache[key] = r

        logging.info("cache built")
        logging.info("building dataframe")
        df = pd.DataFrame.from_records(list(cache.values()), columns=flat_fieldmap.names)
        logging.info("dataframe built")
        df = df[df['ctpdate'] > 19700101]
        df.index = make_time(df['ctpdate'],
                             df['updatetime'])  # actionday is the actual date the trade happened, except for DCE
        # df.columns = map(str.lower, df.columns) ## diff with stock
        # df['ticker'] = df["market"] + df.pop('code') # ticker is prefixed ## diff with stock
        wfields = ['open', 'high', 'low', 'close', 'turnover', 'settle']
        df[wfields] = df[wfields].applymap(lambda x: x / 10000.0)
        ctpdate = df['ctpdate']  # this is tradingday
        df = df[['ticker', 'open', 'high', 'low', 'close', 'turnover', 'volume', 'openint', 'settle']]
        df['date'] = ctpdate
        df['openinterest'] = df.pop('openint')
        df = df[(df.open != 0) & (df.volume != 0)]  ## diff with stock
        return df[['date', 'ticker', 'open', 'high', 'low', 'close', 'turnover', 'volume', 'openinterest',
                   'settle']]  # re-order fields, accoridng to futdaily table

    def make_bars(self, hqxzroot, date, min1root):
        assert os.path.exists(min1root), "path not exist: " + min1root
        logging.info("build future bars")
        flat_fieldmap = ctpfields.future.flat_fieldmap
        records = read_buffer(hqxzroot, date, flat_fieldmap)
        records = records[
            ['ticker', 'last', 'volume', 'turnover', 'openint', 'actionday', 'localdate', 'updatetime', 'localtime']]
        tickers = set(records['ticker'])
        logging.debug("total %s tickers", len(tickers))
        ### NOTE: generator by tickers is faster than pd.groupby operation which takes amount of memory
        for ticker in tickers:
            # see ticker is byte type in python 3
            df = pd.DataFrame.from_records(records[records['ticker'] == ticker])
            df = df[df.volume > 0]  # make sure turnover/openint/volume is not 0
            not_market_close_time = ~(
                    ((df['localtime'] > 160000000) & (df['localtime'] < 205500000))
                    |
                    ((df['localtime'] > 30000000) & (df['localtime'] < 85500000))
            )
            df = df[(df['actionday'] > 19700101) & not_market_close_time]
            if df.empty:
                logging.debug(f"empty tickdata: {date}, {ticker}")
                continue
             
            market = guess_exchange_from_ticker(ticker, date)
            market = market.upper()
            idate = df['actionday']
            if market == "DCE":
                # DCE has no cross 24:00 trading, so it is ok to merge localdate with updatetime
                logging.debug("got DCE market: %s", market)
                idate = df['localdate']
            df.index = make_time(idate, df['updatetime'])  # actionday is the actual date the trade happened
            df = pd.concat([df.iloc[df.index.indexer_between_time('20:59:00', '02:30:59')],
                            df.iloc[df.index.indexer_between_time('8:59:00', '15:15:59')]])
            if df.empty:
                logging.debug(f"no data in legal bars interval: {date}, {ticker}")
                continue
            # do resample
            wfields = ['last', 'turnover']
            for field in wfields:
                df[field] /= 1e4  # here 'last' dtype change from uint4 => float8
            df['size'] = df['openint']
            ohlc = df['last'].resample('T', label='right', closed='right').apply('ohlc')
            # thanks to resample, the u4 dtype is changed to float64
            # otherwise, u4 will make error when diff()-ed later
            bars = df.resample('T', label='right', closed='right').apply({'turnover': 'last','volume': 'last','openint': 'last','size': 'last', }, fill_method='ffill')
            bars = pd.concat([ohlc, bars], axis=1).dropna()
            for col in ['turnover', 'volume', 'size']:
                diff = bars[col].diff()
                assert diff.shape[0] - diff.count() == 1, 'more than one NaN value!'
                diff.iloc[0] = bars[col].iloc[0]  # fill the first NaN value
                bars[col] = diff
                bars = bars.astype({'close': 'f4',
                                    'high': 'f4',
                                    'low': 'f4',
                                    'open': 'f4',
                                    'openint': 'i4',
                                    'size': 'i4',
                                    'turnover': 'f8',
                                    'volume': 'i4'})  # in db, voluem is bigint, but in .hq it is u4
            # bars['ticker'] = df.ticker.iloc[0]
            bars = bars[['open', 'high', 'low', 'close', 'turnover', 'volume', 'openint', 'size']]
            bars = bars[bars.volume > 0]
            csvfn = os.path.join(min1root, bytes(ticker).decode() + ".csv.gz")  # bug prone ticker = bytes(ticker).decode()
            bars.to_csv(csvfn,
                        header=True, index=True, index_label="datetime",
                        encoding="utf-8", compression="gzip")
            logging.debug("write {fn} done".format(fn=csvfn))


    def dump_daily(self, dailydf):
        dailydf.date = dailydf.date.apply(str)  # database type constraint
        logging.info("dump to postgre %s", make_engine())
        # 此处，分开期权和期货
        optpatten = "[a-zA-Z]+[0-9]+-?[CP]-?\d+"
        options = dailydf[dailydf.ticker.str.match(optpatten)]
        futures = dailydf[~ dailydf.ticker.str.match(optpatten)]
        if "allow-future" in sys.argv or \
                "only-future" in sys.argv or \
                "only-option" not in sys.argv:
            futures.to_sql('futdaily', make_engine(), if_exists='append', index_label='timestamp')
        if "allow-option" in sys.argv or \
                "only-option" in sys.argv or \
                "only-future" not in sys.argv:
            options.to_sql('optdaily', make_engine(), if_exists='append', index_label='timestamp')
        logging.info("dumped")
        lates = dailydf[(dailydf.index.time < pd.datetime(1970, 1, 1, 14, 55).time())]
        if not lates.empty:
            email_alarm(lates, "hhfixup报警: future daily price generation")


def makedirs(path):
    os.system("""mkdir -p "%s" """ % path)


if __name__ == "__main__":
    import argparse
    import datetime
    import sys

    parser = argparse.ArgumentParser(
        description='create future daily and future bars from hq file generated by mdserver.ctp.', add_help=True)
    parser.add_argument('--instr', required=True, type=str,
                        choices=["stock", "index", "future", "option", "comm-option", ],
                        help='which instrument type to process')
    parser.add_argument('--data', required=True, dest='data', choices=['daily', 'k'],
                        help='which data type to generate')
    parser.add_argument('--hqroot', required=True, metavar='HQXZ_ROOT', type=str, help='root of hqxz file')
    parser.add_argument('--h5root', metavar='TICKDATA_ROOT', type=str, help='root of H5 tickdata.')
    parser.add_argument('--min1root', metavar='mint1root', type=str, help='root to min1.')
    parser.add_argument('--ctp-instruments-root', metavar='ctp_instruments_root',
                        default="/dat/raw/futures/instruments", type=str, help='root of instrument infomation files')
    parser.add_argument('--date', dest='date', default=None, help='which date to process')
    parser.add_argument('--mode', default="runonce", choices=['scan', 'runonce'])
    #    parser.add_argument('--mailto', default=('whc'), help='csv, a mail list to inform error message')
    args, unknown = parser.parse_known_args()  # to support non-listed allow-future/allow-option/only-future/only-option
    ctp_instruments_root = args.ctp_instruments_root
    args.date = args.date if args.date else datetime.date.today().strftime("%Y%m%d")
    # if args.instr == 'stock':
    #     fixup = StockFixup()
    # elif args.instr == 'index':
    #     fixup = IndexFixup()

    if not os.path.exists(args.hqroot):
        logging.info("create hqroot %s", args.hqroot)
        makedirs(args.hqroot)

    if args.data == "k":
        if not args.min1root:
            logging.warning("must provide --min1root for --data k type")
            sys.exit(1)

    if args.min1root and not os.path.exists(args.min1root):
        logging.info("create min1 root %s", args.min1root)
        makedirs(args.min1root)

    if args.instr == 'future':
        fixup = FutureFixup(args)
    elif args.instr == 'option':
        fixup = OptionFixup()
    else:
        raise Exception("bad argument")

    if args.data == "daily":
        for date in dates:
            try:
                dailydf = fixup.make_daily(args.hqroot, args.h5root, date)
                fixup.dump_daily(dailydf)
            except Exception as e:
                traceback.print_exc(e)
    elif args.data == "k":
        dates = []
        if args.mode == "runonce":
            dates = [args.date, ]
        elif args.mode == "scan":
            import glob
            import re

            pattern = os.path.join(args.hqroot, r"(\d{8}).hq(.xz)?")
            files = glob.glob(os.path.join(args.hqroot, "*.hq")) + glob.glob(os.path.join(args.hqroot, "*.hq.xz"))
            for fn in files:
                if not re.match(pattern, fn):
                    logging.warning("file pattern mismatch: %s !!!", fn)
            files = [fn for fn in files if re.match(pattern, fn)]
            dates = [re.match(pattern, fn).group(1) for fn in files]
            dates = np.unique(dates)
            dates.sort()
        for date in dates:
            if hasattr(guess_exchange_from_ticker, "exchmap"):
                del guess_exchange_from_ticker.exchmap
            assert not hasattr(guess_exchange_from_ticker, "exchmap")
            start_time = pd.Timestamp.now()
            logging.debug("process date %s", date)
            dayroot = os.path.join(args.min1root, date)
            if os.path.exists(dayroot):
                logging.warning(f"{dayroot} exists, ignore task.")
                continue
            makedirs(dayroot)
            fixup.make_bars(args.hqroot, date, dayroot)
            finish_time = pd.Timestamp.now()
            logging.info("chmod to make folder readonly")
            os.system(f"""chmod -R a-w "{dayroot}" """)
            logging.info("performance. time cost %s", finish_time - start_time)
