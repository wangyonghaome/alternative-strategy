#!/home/ywang/env/dev/bin
#-*-coding:utf-8-*-
#written by wangyonghao

#the api: get_crosscount_summary(date,period_ls)

#input: date: current date ( "20190102" for example)
#       period_ls:the list for the window period ([5,10,15] for example)


#output to file: the excel file of the cross_grid_count_summary for each window_period and for each stock



import numpy as np
import pandas as pd
import tushare as ts
from fb_base_get_market_data import *
from data_tools.api import *
from utilscht.Data import *
import click
import logging

logging.basicConfig(level=logging.DEBUG,#控制台打印的日志级别
                    filename=r'logging/crosscount_summary.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'#日志格式
                    )

date_ls = trade_days

def factor_data_calcu(td_date, stock_ls, window_period, factor_df, crosscount_df):
    date_idx = date_ls.index(td_date)
    date_range = date_ls[date_idx - window_period + 1:date_idx + 1]

    count = np.sum(np.array(crosscount_df.loc[date_range, stock_ls]), axis=0)

    factor_df.loc[stock_ls, "{}d_count".format(window_period)] = count

@click.command()
@click.argument("date",nargs=1)
@click.argument("period_ls",nargs=-1)
def get_crosscount_summary(date,period_ls):
    period_ls=[int(i) for i in period_ls]
    date_range = date_ls[max(1, date_ls.index(date) - 29):date_ls.index(date) + 1]
    crosscount_df = pd.DataFrame()
    for datadate in date_range:
        temp_df = pd.read_csv(r"/home/ywang/proj_cross_grid/result/cross_count_b240/crosscount_{}.csv".format(datadate), index_col="sid")
        temp_df = temp_df.reindex(pd.MultiIndex.from_product([[datadate], temp_df.index]), level=1)
        crosscount_df = pd.concat([crosscount_df, temp_df])
    crosscount_df = crosscount_df.unstack()["cross_count"]
    crosscount_df = crosscount_df.dropna(axis=1, how="any")

    factor_df = pd.DataFrame(index=crosscount_df.columns,
                             columns=[str(i)+"d_count" for i in period_ls])

    for window_period in period_ls:
        factor_data_calcu(date, crosscount_df.columns, window_period, factor_df, crosscount_df)
    factor_df["mean_5-mean_30"]=factor_df["5d_count"]/5-factor_df["30d_count"]/30
    factor_df["mean_10-mean_30"]=factor_df["10d_count"]/10-factor_df["30d_count"]/30
    
    pre_date = get_previous_trade_date(date)
    df_mktv = get_db(start_date= pre_date, end_date=pre_date,cols = ["market_value"])\
                    .drop(["date"],axis =1).set_index("sid")
    factor_df["market_value"] = df_mktv["market_value"]
    
    
    factor_df.to_excel(r"/home/ywang/proj_cross_grid/result/cross_count_b240/crosscount_summary_{}.xlsx"\
                       .format(date),index_label = "sid")
    
    pre_date = get_previous_trade_date(date)
    stock_pool=pd.read_csv(r"/share/xfzhang/to_colleague/to_yzhao/task2/{}/{}/task2_{}.csv".\
                                        format(pre_date[0:4],pre_date[4:6],pre_date[0:8]))["sid"]
    factor_df_baima=factor_df.loc[stock_pool]
    factor_df_baima.to_excel(r"/home/ywang/proj_cross_grid/result/cross_count_b240_baima/crosscount_summary_{}.xlsx"\
                             .format(date),index_label="sid")
                       
    logging.info("{} finished".format(date))

if __name__ == '__main__':
    get_crosscount_summary()