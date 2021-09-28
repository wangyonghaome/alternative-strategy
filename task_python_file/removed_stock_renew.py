#!/home/ywang/env/dev/bin
#-*-coding:utf-8-*-
# written by wangyonghao

"""
统计每次更新底仓时被移除的股票
"""
import numpy as np
import pandas as pd
from data_tools.api import *
from utilscht.Data import *
from fb_base_calendar_description import *
import pymysql
import datetime


date = str(datetime.datetime.now().date()).replace('-','')


filename_1,filename_2 = os.listdir("/share/xfzhang/to_colleague/to_yzhao/task_stock_pool/{}/".format(date[0:4]))[-2:]
filename_1 =filename_1.replace("stock_pool","task1")
filename_2 =filename_2.replace("stock_pool","task1")

stock_pool_last = pd.read_csv(r"/share/xfzhang/to_colleague/to_yzhao/task1/{}/{}/{}".
                                format(filename_1[6:10],filename_1[10:12],filename_1),dtype={"date":str})["sid"]
stock_pool_now = pd.read_csv(r"/share/xfzhang/to_colleague/to_yzhao/task1/{}/{}/{}".
                                format(filename_2[6:10],filename_2[10:12],filename_2),dtype={"date":str})["sid"]
stock_pool_removed = list(set(stock_pool_last)- set(stock_pool_now))

df_removed = pd.DataFrame(data=stock_pool_removed,columns=["sid"])
df_removed["date"] = date
df_removed.to_excel(r"result/removed_stock/removed_stock_baima_{}.xlsx".format(date),index=False)