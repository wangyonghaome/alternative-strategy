#!/home/ywang/env/dev/bin
#-*-coding:utf-8-*-
# written by wangyonghao

# the api: get_daily_cross_data(date)

# input:  date: current date ( "20190102" for example)

# output to file: the csv file of the cross_grid_count for each stock

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from data_tools.api import *
from utilscht.Data import *
import datetime


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


def get_volatility(arr, th=0.010):
    
    if len(arr[arr<=0.0])>0:
        print(arr[arr<=0.0])
        return np.nan
    m = arr.shape[0]
    log_close = np.log(arr)
    volat = np.zeros_like(log_close)
    
    state = 0 # initial
    node, node2 = 0, 0
    node_min, node_max = 0, 0
    for i in range(1, m):
        if state == 0:
            if log_close[i] < log_close[node_min]:
                node_min = i
            if log_close[i] > log_close[node_max]:
                node_max = i
            if log_close[node_max] - log_close[node_min] > th:
                if node_max > node_min:
                    node = node_min
                    node2 = node_max
                    state = 1
                elif node_max < node_min:
                    node = node_max
                    node2 = node_min
                    state = -1
        elif state == 1:
            if log_close[i] > log_close[node2]:
                node2 = i
            elif log_close[i] - log_close[node2] < -th:
                node = node2
                node2 = i
                state = -1
        elif state == -1:
            if log_close[i] < log_close[node2]:
                node2 = i
            elif log_close[i] - log_close[node2] > th:
                node = node2
                node2 = i
                state = 1
        volat[i] = volat[node] + abs(log_close[node2] - log_close[node])
    
    return volat[-1]

