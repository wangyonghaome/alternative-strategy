import numpy as np
import pandas as pd
import statsmodels.api as sm
import pymysql

DB_INFO = dict(host='192.168.1.234',
               user='winduser',
               password='1qaz@WSX',
               db='wind')

from data_tools.api import trade_days
trade_dates_all = trade_days.copy()
def get_prev_n_trade_date(trade_date, n):
    pos = np.searchsorted(trade_dates_all, trade_date)
    assert pos >= n
    return int(trade_dates_all[pos - n])

def get_next_n_trade_date(trade_date, n=1):
    pos = np.searchsorted(trade_dates_all, trade_date, side='right')
    assert pos + n - 1 < len(trade_dates_all)
    return int(trade_dates_all[pos + n - 1])


#公司代码和股票代码对照表
def get_code_map(conn):
    sql="SELECT S_INFO_WINDCODE, S_INFO_COMPCODE from WINDCUSTOMCODE where S_INFO_SECURITIESTYPES='A'"
    print('Reading sql WindCustomCode')
    df_code_map = pd.read_sql_query(sql,conn)
    df_code_map.rename({'S_INFO_WINDCODE': 'sid', 'S_INFO_COMPCODE': 's_info_compcode'}, axis=1, inplace=True)
    df_code_map.sort_values('sid', inplace=True)
    assert len(np.unique(df_code_map.sid)) == len(df_code_map)
    code_map = {ele['s_info_compcode']: ele['sid'] for _, ele in df_code_map.iterrows()}
    return code_map


###### Part 1 ######
# 应计利润，操纵性应计利润、真实盈余管理
def get_accural_info(date, conn):
    date_p2y = get_prev_n_trade_date(date, 480)
    date_p3y = get_prev_n_trade_date(date, 720)
    
    # 获取行业
    sql = "SELECT S_INFO_WINDCODE, ENTRY_DT, CITICS_IND_CODE from ASHAREINDUSTRIESCLASSCITICS where ENTRY_DT<={}".format(date)
    print('Reading sql AShareIndustriesClassCITICS')
    df_indus = pd.read_sql_query(sql, conn)
    df_indus.rename({'S_INFO_WINDCODE': 'sid', 'ENTRY_DT': 'entry_dt', 'CITICS_IND_CODE': 'industry'}, axis=1, inplace=True)
    df_indus['entry_dt'] = df_indus['entry_dt'].values.astype('i4')
    df_indus = df_indus[df_indus.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_indus.sort_values(by=['sid', 'entry_dt'], inplace=True)
    sr_indus = df_indus.groupby('sid')['industry'].last().astype('S10')
    sr_indus = 1.0 * sr_indus.apply(lambda x: x[3] - 48 if x[3] <= 57 else x[3] - 87)  # 1.0~9.0  10.0~29.0

    sql = ("""select S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, TOT_ASSETS, FIX_ASSETS from ASHAREBALANCESHEET"""
           + """ where STATEMENT_TYPE ='408001000' and ANN_DT<='{}' and REPORT_PERIOD>='{}' and REPORT_PERIOD like '%1231'""".format(date, date_p3y) )
    print('Reading sql AShareBalanceSheet')
    df_balance = pd.read_sql_query(sql, conn)
    df_balance.columns = [ele.lower() for ele in df_balance.columns]
    df_balance.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_balance.ann_dt = df_balance.ann_dt.astype('i4')
    df_balance.report_period = df_balance.report_period.astype('i4')
    df_balance = df_balance[df_balance.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))] # & (~np.isnan(df_balance.tot_assets))]
    df_balance.sort_values(['sid', 'report_period', 'ann_dt'], inplace=True)
    
    sql = ("""select S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, OPER_REV, OPER_PROFIT, LESS_OPER_COST, LESS_SELLING_DIST_EXP, LESS_GERL_ADMIN_EXP from ASHAREINCOME"""
           + """ where STATEMENT_TYPE ='408001000' and ANN_DT<='{}' and REPORT_PERIOD>='{}' and REPORT_PERIOD like '%1231'""".format(date, date_p2y) )
    print('Reading sql AShareIncome')
    df_income = pd.read_sql_query(sql, conn)
    df_income.columns = [ele.lower() for ele in df_income.columns]
    df_income.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_income.ann_dt = df_income.ann_dt.astype('i4')
    df_income.report_period = df_income.report_period.astype('i4')
    df_income = df_income[df_income.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_income.sort_values(['sid', 'report_period', 'ann_dt'], inplace=True)
    
    sql = ("""select S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, STOT_CASH_INFLOWS_OPER_ACT, NET_CASH_FLOWS_OPER_ACT from ASHARECASHFLOW"""
           + """ where STATEMENT_TYPE ='408001000' and ANN_DT<='{}' and REPORT_PERIOD>='{}' and REPORT_PERIOD like '%1231'""".format(date, date_p2y) )
    print('Reading sql AShareCashFlow')
    df_cash = pd.read_sql_query(sql, conn)
    df_cash.columns = [ele.lower() for ele in df_cash.columns]
    df_cash.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_cash.ann_dt = df_cash.ann_dt.astype('i4')
    df_cash.report_period = df_cash.report_period.astype('i4')
    df_cash = df_cash[df_cash.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_cash.sort_values(['sid', 'report_period', 'ann_dt'], inplace=True)
    
    # 计算，需要的列
    sr_tot_assets_pre = df_balance.groupby("sid")["tot_assets"].apply(lambda x: x.iloc[-2] if len(x)>1 else np.nan)
    sr_oper_rev_delta = df_income.groupby("sid")['oper_rev'].apply(lambda x: x.iloc[-1]-x.iloc[-2] if len(x)>1 else np.nan)
    sr_oper_rev_delta.name = 'oper_rev_delta'
    
    sr_fix_assets = df_balance.groupby("sid")['fix_assets'].last()
    df_income2 = df_income.groupby("sid")[['oper_rev', 'oper_profit', 'less_oper_cost', 'less_selling_dist_exp', 'less_gerl_admin_exp']].last()
    df_cash2 = df_cash.groupby("sid")[['stot_cash_inflows_oper_act', 'net_cash_flows_oper_act']].last()
#     sr_oper_profit = df_income.groupby("sid")["oper_profit"].last()
#     sr_net_oper_cashflow = df_cash.groupby("sid")["net_cash_flows_oper_act"].last()
    df_summary = pd.concat([sr_tot_assets_pre, sr_oper_rev_delta, sr_fix_assets, df_income2, df_cash2], axis=1, sort=True)
    df_summary.index.name = 'sid'
    df_summary = df_summary.merge(sr_indus, on='sid', how='left')
    
    # 应计利润
    df_summary['accural_profit_ratio'] = (df_summary.oper_profit - df_summary.net_cash_flows_oper_act) / (np.abs(df_summary.tot_assets) + 1.e4)
    sr_accural_profit =df_summary.accural_profit_ratio
    
    # 操纵性应计利润
    def func_get_resid(df, cols_x, col_y):
        df = df[cols_x + [col_y]].dropna()
        x = df[cols_x].values
        y = df[col_y].values
        if len(x) <= 3:
            return pd.Series([])
        x=sm.add_constant(x)
        result=sm.OLS(y, x).fit()
        return pd.Series(result.resid, index=df.index.get_level_values('sid'))
    df_summary['oper_rev_delta_ratio'] = df_summary.oper_rev_delta / (np.abs(df_summary.tot_assets) + 1.e4)
    df_summary['fix_assets_ratio'] = df_summary.fix_assets / (np.abs(df_summary.tot_assets) + 1.e4)
    sr_accural_abnormal = df_summary.groupby('industry', as_index=False).apply(
        lambda x: func_get_resid(x, ['oper_rev_delta_ratio', 'fix_assets_ratio'], 'accural_profit_ratio')).reset_index(0, drop=True)
    sr_accural_abnormal.sort_index(inplace=True)
    
    # 真实盈余管理
    for col in ['oper_rev', 'stot_cash_inflows_oper_act', 'less_oper_cost', 'less_selling_dist_exp', 'less_gerl_admin_exp']:  # 营业收入变化之前取过ratio
        df_summary[col+'_ratio'] = df_summary[col] / (np.abs(df_summary.tot_assets) + 1.e4)
    sr1 = -df_summary.groupby('industry', as_index=False).apply(
        lambda x: func_get_resid(x, ['oper_rev_ratio', 'oper_rev_delta_ratio'], 'stot_cash_inflows_oper_act_ratio')).reset_index(0, drop=True)
    sr2 = df_summary.groupby('industry', as_index=False).apply(
        lambda x: func_get_resid(x, ['oper_rev_ratio', 'oper_rev_delta_ratio'], 'less_oper_cost_ratio')).reset_index(0, drop=True)
    df_summary['sell_admin_ratio'] = df_summary.less_selling_dist_exp_ratio + df_summary.less_gerl_admin_exp_ratio
    sr3 = -df_summary.groupby('industry', as_index=False).apply(
        lambda x: func_get_resid(x, ['oper_rev_ratio'], 'sell_admin_ratio')).reset_index(0, drop=True)
    sr_manipulate = sr1 + sr2 + sr3
    sr_manipulate.sort_index(inplace=True)
    
    sr_accural_profit.name ='accural_profit'
    sr_accural_abnormal.name = 'accural_abnormal'
    sr_manipulate.name = 'manipulate'
    return sr_accural_profit, sr_accural_abnormal, sr_manipulate
    

## 在建工程增速、在建工程与现金流不匹配度 (回归系数的相反数感觉有点问题，这样的话每个行业也就一个值)
def get_construct_info(date, conn):
    date_p2y = get_prev_n_trade_date(date, 480)
    date_p3y = get_prev_n_trade_date(date, 720)
    
    # 获取行业
    sql = "SELECT S_INFO_WINDCODE, ENTRY_DT, CITICS_IND_CODE from ASHAREINDUSTRIESCLASSCITICS where ENTRY_DT<={}".format(date)
    print('Reading sql AShareIndustriesClassCITICS')
    df_indus = pd.read_sql_query(sql, conn)
    df_indus.rename({'S_INFO_WINDCODE': 'sid', 'ENTRY_DT': 'entry_dt', 'CITICS_IND_CODE': 'industry'}, axis=1, inplace=True)
    df_indus['entry_dt'] = df_indus['entry_dt'].values.astype('i4')
    df_indus = df_indus[df_indus.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_indus.sort_values(by=['sid', 'entry_dt'], inplace=True)
    sr_indus = df_indus.groupby('sid')['industry'].last().astype('S10')
    sr_indus = 1.0 * sr_indus.apply(lambda x: x[3] - 48 if x[3] <= 57 else x[3] - 87)  # 1.0~9.0  10.0~29.0
    
    sql = ("""select S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, FIX_ASSETS, CONST_IN_PROG from ASHAREBALANCESHEET """
           + """ where STATEMENT_TYPE ='408001000' and ANN_DT<='{}' and REPORT_PERIOD>='{}'""".format(date, date_p3y)
           + """ and (REPORT_PERIOD like '%0630' or REPORT_PERIOD like '%1231')""")
    print('Reading sql AShareBalanceSheet')
    df_balance = pd.read_sql_query(sql,conn)
    df_balance.columns = [ele.lower() for ele in df_balance.columns]
    df_balance.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_balance.ann_dt = df_balance.ann_dt.astype('i4')
    df_balance.report_period = df_balance.report_period.astype('i4')
    df_balance = df_balance[df_balance.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_balance.sort_values(['sid', 'report_period', 'ann_dt'], inplace=True)
    
    sql = ("""select S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, CASH_PAY_ACQ_CONST_FIOLTA from ASHARECASHFLOW"""
           """ where STATEMENT_TYPE ='408001000' and ANN_DT<='{}' and REPORT_PERIOD>='{}' and REPORT_PERIOD like '%1231'""".format(date, date_p2y))
    print('Reading sql AShareCashFlow')
    df_cash = pd.read_sql_query(sql,conn)
    df_cash.columns = [ele.lower() for ele in df_cash.columns]
    df_cash.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_cash.ann_dt = df_cash.ann_dt.astype('i4')
    df_cash.report_period = df_cash.report_period.astype('i4')
    df_cash = df_cash[df_cash.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_cash.sort_values(['sid', 'report_period', 'ann_dt'], inplace=True)
    
    # 在建工程增速
    def func_get_growth(df):
        df = df[['sid', 'report_period', 'const_in_prog']].dropna()
        x = df['report_period'].values//10000 + df['report_period'].values%10000//100/12.0
        y = df.const_in_prog.values
        if len(x) < 3:
            return np.nan
        y_mean = np.mean(y)
        if y_mean == 0.0:
            return np.nan
        x = x - np.mean(x)
        y2 = y - y_mean
        beta = np.sum(y2 * x) / np.sum(x * x)
        return beta / np.abs(y_mean)
    sr_construct_growth = df_balance.groupby("sid").apply(func_get_growth)
    
    # 在建工程与现金流不匹配度
    def func_get_coef(df, cols_x, col_y, idx):  # 回归系数的相反数感觉有点问题，这样的话每个行业也就一个值
        df = df[cols_x + [col_y]].dropna()
        x = df[cols_x].values
        y = df[col_y].values
        if len(x) <= 3:
            return pd.Series([])
        x=sm.add_constant(x)
        result=sm.OLS(y, x).fit()
        return pd.Series(result.params[idx+1], index=df.index.get_level_values('sid'))
    df_balance2 = df_balance[df_balance.report_period % 10000 == 1231]
    sr_fix_assets_delta = df_balance2.groupby("sid")['fix_assets'].apply(lambda x: x.iloc[-1]-x.iloc[-2] if len(x)>1 else np.nan)
    sr_fix_assets_delta.name = 'fix_assets_delta'
    sr_construct_delta = df_balance2.groupby("sid")['const_in_prog'].apply(lambda x: x.iloc[-1]-x.iloc[-2] if len(x)>1 else np.nan)
    sr_construct_delta.name = 'construct_delta'
    sr_cash_pay_acq = df_cash.groupby("sid")['cash_pay_acq_const_fiolta'].last()
    df_summary = pd.concat([sr_fix_assets_delta, sr_construct_delta, sr_cash_pay_acq], axis=1, sort=True)
    df_summary.index.name = 'sid'
    df_summary = df_summary.merge(sr_indus, on='sid', how='left')
    sr_construct_coef = -df_summary.groupby('industry', as_index=False).apply(
        lambda x: func_get_coef(x, ['fix_assets_delta', 'construct_delta'], 'cash_pay_acq_const_fiolta', 1)).reset_index(0, drop=True)
    sr_construct_coef.sort_index(inplace=True)
    
    sr_construct_growth.name = 'construct_growth'
    sr_construct_coef.name = 'construct_coef'
    return sr_construct_growth, sr_construct_coef * np.nan
    
    
## 研发资本化、会计师事务所变更
def get_rd_account_info(date, conn):
    date_p2y = get_prev_n_trade_date(date, 480)
    date_p3y = get_prev_n_trade_date(date, 720)
    code_map = get_code_map(conn)
    
    sql = ("""select S_INFO_COMPCODE, ANN_DT, REPORT_PERIOD, STATEMENT_TYPE, ANN_ITEM, ITEM_AMOUNT from ASHARERDEXPENDITURE"""
           + """ where STATEMENT_TYPE='合并报表' and ANN_DT<='{}' and REPORT_PERIOD>='{}' and REPORT_PERIOD like '%1231'""".format(date, date_p2y))
    print('Reading sql AShareRDexpenditure')
    df_rd_expend = pd.read_sql_query(sql,conn)
    df_rd_expend.columns = [ele.lower() for ele in df_rd_expend.columns]
    df_rd_expend.ann_dt = df_rd_expend.ann_dt.astype('i4')
    df_rd_expend.report_period = df_rd_expend.report_period.astype('i4')
    df_rd_expend['sid'] = [code_map.get(ele, '0') for ele in df_rd_expend.s_info_compcode]
    df_rd_expend = df_rd_expend[df_rd_expend.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_rd_expend.sort_values(['sid', 'report_period', 'ann_dt'], inplace=True)
    
    sql= ("""select S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, RD_EXPENSE from ASHAREINCOME"""
             + """ where STATEMENT_TYPE ='408001000' and ANN_DT<={} and REPORT_PERIOD>='{}' and REPORT_PERIOD like '%1231'""".format(date, date_p2y))
    print('Reading sql AShareIncome')
    df_income = pd.read_sql_query(sql,conn)
    df_income.columns = [ele.lower() for ele in df_income.columns]
    df_income.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_income.ann_dt = df_income.ann_dt.astype('i4')
    df_income.report_period = df_income.report_period.astype('i4')
    df_income = df_income[df_income.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_income.sort_values(['sid', 'report_period', 'ann_dt'], inplace=True)
    
    #计算 研发比率
    def func_calc_capitalize_expense(df):
        last_period = df['report_period'].iloc[-1]
        df = df[df.report_period == last_period]
        capitalize_items = ['本期资本化研发投入', '本期资本化研发支出', '研发投入资本化的金额(元)', '研发支出资本化的金额(元)']
        df = df[df.ann_item.isin(capitalize_items)]
        return df.item_amount.sum()
    sr_rd_capitalize = df_rd_expend.groupby('sid').apply(func_calc_capitalize_expense)
    sr_rd_expense = df_income.groupby('sid')['rd_expense'].last()
    sr_rd_ratio = sr_rd_capitalize / (np.abs(sr_rd_expense) + 1.e4)
    
    # 会计师事务所
    sql = ("""select S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, S_STMNOTE_AUDIT_AGENCY from ASHAREAUDITOPINION"""
           + """ where ANN_DT<='{}' and REPORT_PERIOD>='{}'""".format(date, date_p3y))
    print('Reading sql AShareAuditOpinion')
    df_audit = pd.read_sql_query(sql,conn)
    df_audit.columns = [ele.lower() for ele in df_audit.columns]
    df_audit.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_audit.ann_dt = df_audit.ann_dt.astype('i4')
    df_audit.report_period = df_audit.report_period.astype('i4')
    df_audit = df_audit[df_audit.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_audit.sort_values(['sid', 'report_period', 'ann_dt'], inplace=True)
    # 计算
    sr_change_audit = 1.0 * df_audit.groupby('sid')['s_stmnote_audit_agency'].apply(lambda x: len(np.unique(x))>=2)
    
    sr_rd_ratio.name = 'rd_ratio'
    sr_change_audit.name = 'change_audit'
    return sr_rd_ratio, sr_change_audit 
    
    
## 质押占比、报告期质押冻结、信托占比、信托数量
def get_capital_risk_info(date, conn):
    date_p1m = get_prev_n_trade_date(date, 20)
    date_p1y = get_prev_n_trade_date(date, 240)
    
    sql = ("""select S_INFO_WINDCODE, S_ENDDATE, S_PLEDGE_RATIO from ASHAREPLEDGEPROPORTION"""
           + """ where S_ENDDATE<='{}' and S_ENDDATE>='{}'""".format(date, date_p1m))
    print('Reading sql ASharePledgeproportion')
    df_pledge = pd.read_sql_query(sql,conn)
    df_pledge.columns = [ele.lower() for ele in df_pledge.columns]
    df_pledge.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_pledge.s_enddate = df_pledge.s_enddate.astype('i4')
    df_pledge = df_pledge[df_pledge.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_pledge.sort_values(['sid', 's_enddate'], inplace=True)
    # 质押比例还是从ASharePledgeproportion拿吧，ann_dt为每周五，但周一凌晨才会更新
    sr_pledge_ratio = 0.01 * df_pledge.groupby('sid')['s_pledge_ratio'].last()
    
    # 先获取总股本，原始单位为万股
    sql = """select S_INFO_WINDCODE, CHANGE_DT, TOT_SHR from ASHARECAPITALIZATION where CHANGE_DT<={}""".format(date)
    print('Reading sql AShareCapitalization')
    df_tot_share = pd.read_sql_query(sql, conn)
    df_tot_share.rename({'S_INFO_WINDCODE': 'sid', 'CHANGE_DT': 'change_dt', 'TOT_SHR': 'tot_shr'}, axis=1, inplace=True)
    df_tot_share['change_dt'] = df_tot_share['change_dt'].values.astype('i4')
    df_tot_share = df_tot_share[df_tot_share.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_tot_share.sort_values(by=['sid', 'change_dt'], inplace=True)
    sr_tot_share = 10000.0 * df_tot_share.groupby('sid')['tot_shr'].last()
    ## 中国A股股权冻结质押情况(报告期)中 股票单位都是股
    sql=("""select S_INFO_WINDCODE, F_NAV_UNIT, PRICE_DATE, F_NAV_DIVACCUMULATED from AEQUFROPLEINFOREPPEREND"""  # F_NAV_UNIT公告日期，PRICE_DATE报告期
           + """ where F_NAV_UNIT<='{}' and PRICE_DATE>='{}'""".format(date, date_p1y))
    print('Reading sql AEquFroPleInfoRepperend')
    df_fro_ple = pd.read_sql_query(sql,conn)
    df_fro_ple.columns = [ele.lower() for ele in df_fro_ple.columns]
    df_fro_ple.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_fro_ple.f_nav_unit = df_fro_ple.f_nav_unit.astype('i4')
    df_fro_ple.price_date = df_fro_ple.price_date.astype('i4')
    df_fro_ple = df_fro_ple[df_fro_ple.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_fro_ple.sort_values(['sid', 'price_date', 'f_nav_unit'], inplace=True)
    # 计算
    def func_calc_fro_ple(df):
        last_period = df['price_date'].iloc[-1]
        df = df[df.price_date == last_period]
        return df.f_nav_divaccumulated.sum()
    sr_fro_ple_share = df_fro_ple.groupby('sid').apply(func_calc_fro_ple)
    sr_fro_ple_ratio = sr_fro_ple_share / sr_tot_share
    
    ## 前十大流通股东 信托占比,单位是股（前十大股东这张表 不一定是流通股东,所以还是应该在流通股东那张表中找）
    sql=("""select S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, S_HOLDER_NAME, S_HOLDER_QUANTITY from ASHAREFLOATHOLDER"""
         + """ where ANN_DT<='{}' and REPORT_PERIOD>='{}'""".format(date, date_p1y))
    print('Reading sql AShareFloatHolder')
    df_float_holder = pd.read_sql_query(sql,conn)
    df_float_holder.columns = [ele.lower() for ele in df_float_holder.columns]
    df_float_holder.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_float_holder.ann_dt = df_float_holder.ann_dt.astype('i4')
    df_float_holder.report_period = df_float_holder.report_period.astype('i4')
    df_float_holder = df_float_holder[df_float_holder.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_float_holder.sort_values(['sid', 'report_period', 'ann_dt'], inplace=True)
    # 计算
    def func_float_top10(df):
        last_period = df["report_period"].iloc[-1]
        df = df[df["report_period"]==last_period]
        df.sort_values("s_holder_quantity", ascending=False, inplace=True)
        df = df.iloc[:10]
        df["is_trust"] = 1.0 * df["s_holder_name"].apply(lambda x: '信托' in x)
        trust_ratio = np.sum(df["s_holder_quantity"] * df["is_trust"]) / np.sum(df["s_holder_quantity"])
        trust_num = np.sum(df["is_trust"])
        sr = pd.Series([trust_ratio, trust_num], index=["trust_ratio", "trust_num"])
        return sr
    df_top10_result = df_float_holder.groupby('sid').apply(func_float_top10)
    sr_top10_trust_ratio = df_top10_result['trust_ratio']
    sr_top10_trust_num = df_top10_result['trust_num']    

    sr_pledge_ratio.name = 'pledge_ratio'
    sr_fro_ple_ratio.name = 'fro_ple_ratio'
    sr_top10_trust_ratio.name = 'top10_trust_ratio'
    sr_top10_trust_num.name = 'top10_trust_num'
    return sr_pledge_ratio, sr_fro_ple_ratio, sr_top10_trust_ratio, sr_top10_trust_num
    
    
## 近三个月cfo离职、高管离职
def get_manager_leave_info(date, conn):
    date_p3m = get_prev_n_trade_date(date, 60)
    
    #总共11w条记录，有7w条起S_INFO_MANAGER_LEAVEDATE为None
    sql = ("""select S_INFO_WINDCODE, ANN_DATE, S_INFO_MANAGER_LEAVEDATE, S_INFO_MANAGER_TYPE, S_INFO_MANAGER_POST from ASHAREMANAGEMENT"""
           + """ where (ANN_DATE<='{}' and S_INFO_MANAGER_LEAVEDATE>='{}' and S_INFO_MANAGER_LEAVEDATE<='{}')""".format(date, date_p3m, date)
           + """ or (ANN_DATE>='{}' and ANN_DATE<='{}' and S_INFO_MANAGER_LEAVEDATE is null)""".format(date_p3m, date   ))
    print('Reading sql AShareManagement')
    df_management = pd.read_sql_query(sql,conn)
    df_management.columns = [ele.lower() for ele in df_management.columns]
    df_management.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    #df_management.loc[pd.isnull(df_management.s_info_manager_leavedate), 's_info_manager_leavedate'] = df_management.ann_date #wyh:这一步为什么要这么赋值
    #df_management.ann_date = df_management.ann_date.astype('i4')
    #df_management.s_info_manager_leavedate = df_management.s_info_manager_leavedate.astype('i4')
    df_management = df_management[df_management.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_management.sort_values(['sid', 's_info_manager_leavedate'], inplace=True)
    
    #计算
    def func_get_cfo_leave(df):
        df=df[pd.notnull(df.s_info_manager_leavedate)]
        sr_cfo_leave = df["s_info_manager_post"].apply(lambda x: x in ['总会计师', '财务总监', '首席财务官', '首席财务长'])  #zxf:这五者有点不准确
        cfo_leave = 1.0 * (np.sum(sr_cfo_leave) > 0)
        return cfo_leave
    def func_get_manager_leave(df):
        sr_manager_leave = (df.s_info_manager_type == 1 & pd.notnull(df.s_info_manager_leavedate))  #高管
        #manager_leave = 1.0 * (np.sum(sr_manager_leave) >= 3)
        manager_leave_ratio = np.sum(sr_manager_leave) / np.sum(df.s_info_manager_type == 1)
        
        return manager_leave_ratio
    sr_cfo_leave = df_management.groupby('sid').apply(func_get_cfo_leave)
    sr_manager_leave = df_management.groupby('sid').apply(func_get_manager_leave)
    
    sr_cfo_leave.name = 'cfo_leave'
    sr_manager_leave.name = 'manager_leave'
    return sr_cfo_leave, sr_manager_leave
    
    
## 财报披露调整, 违规行为， 立案调查， 诉讼仲裁， 业绩预告不准确度， 都是哑变量
def get_inter_control_info(date, conn):
    date_p1y = get_prev_n_trade_date(date, 240)
    
    # 先获取上市日期，在上市日期前发布的都不算逾期
    sql = ("""select S_INFO_WINDCODE, S_INFO_LISTDATE, S_INFO_DELISTDATE from ASHAREDESCRIPTION"""
             + """ where S_INFO_LISTDATE<={} and (S_INFO_DELISTDATE>'{}' or S_INFO_DELISTDATE is null)""".format(date, date))
    print('Reading sql AShareDescription')
    df_descr = pd.read_sql_query(sql, conn)
    df_descr.columns = [ele.lower() for ele in df_descr.columns]
    df_descr.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_descr.s_info_listdate = df_descr.s_info_listdate.astype('i4')
    df_descr = df_descr[df_descr.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_descr.sort_values(['sid', 's_info_listdate'], inplace=True)
    code_list_map = {ele['sid']: ele['s_info_listdate'] for _, ele in df_descr.iterrows()}

    # ASHAREISSUINGDATEPREDICT中国A股定期报告披露日期的 实际披露日期为第一次公告的日期
    # 不知道去哪里可以拿到 调整过财务披露日，AShareIssuingDatePredict groupby(sid,ann_dt).len不行，都为1；其中更正公告披露次数也不对
    sql = ("""select S_INFO_WINDCODE, S_STM_ACTUAL_ISSUINGDATE, REPORT_PERIOD from ASHAREISSUINGDATEPREDICT"""
             + """ where S_STM_ACTUAL_ISSUINGDATE<={} and REPORT_PERIOD>='{}'""".format(date, date_p1y))
    print('Reading sql AShareIssuingDatePredict')
    df_issuing_date = pd.read_sql_query(sql,conn)
    df_issuing_date.columns = [ele.lower() for ele in df_issuing_date.columns]
    df_issuing_date.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_issuing_date.s_stm_actual_issuingdate = df_issuing_date.s_stm_actual_issuingdate.astype('i4')
    df_issuing_date.report_period = df_issuing_date.report_period.astype('i4')
    df_issuing_date = df_issuing_date[df_issuing_date.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_issuing_date.sort_values(['sid', 'report_period', 's_stm_actual_issuingdate'], inplace=True)
    # 处理
    def func_get_due_date(arr):
        year = arr // 10000
        mmdd = arr % 10000
        due_dates = 10000*year + 430*(mmdd==331) + 831*(mmdd==630) + 1031*(mmdd==930) + 10430*(mmdd==1231)
        due_dates = [get_next_n_trade_date(ele) for ele in due_dates]  # 推迟一天吧，有些财报记录会有问题
        return due_dates
    df_issuing_date['due_date'] = func_get_due_date(df_issuing_date.report_period.values)
    df_issuing_date['list_date'] = [code_list_map.get(ele, 0) for ele in df_issuing_date.sid]
    df_issuing_date['is_delay'] = 1.0 * ((df_issuing_date.s_stm_actual_issuingdate > df_issuing_date.due_date) 
                                         & (df_issuing_date.s_stm_actual_issuingdate > df_issuing_date.list_date)) 
    sr_issuing_delay = df_issuing_date.groupby('sid')['is_delay'].sum()
    
    # 违规行为， 这个条件有点宽松,20191101剔除549只股票
    sql = ("""select S_INFO_WINDCODE, ANN_DT from ASHAREILLEGALITY"""
           + """ where RELATION_TYPE='458001000' and ANN_DT<='{}' and ANN_DT>='{}'""".format(date, date_p1y))
    print('Reading sql AShareIllegality')
    df_illegality = pd.read_sql_query(sql, conn)
    df_illegality.rename({'S_INFO_WINDCODE': 'sid'}, axis=1, inplace=True)
    df_illegality = df_illegality[df_illegality.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    sr_illegality = 1.0 * df_illegality.groupby("sid").apply(lambda x:(len(x)))
    
    # 立案调查
    sql = ("""select S_INFO_WINDCODE, STR_DATE from ASHAREREGINV"""
           + """ where STR_DATE<='{}' and STR_DATE>='{}'""".format(date, date_p1y))   
    print('Reading sql AShareRegInv')
    df_reginv=pd.read_sql_query(sql, conn)
    df_reginv.rename({'S_INFO_WINDCODE': 'sid'}, axis=1, inplace=True)
    df_reginv = df_reginv[df_reginv.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    sr_reginv = 1.0 * df_reginv.groupby("sid").apply(lambda x:(len(x)>0))
    
    # 诉讼仲裁
    sql = ("""select S_INFO_WINDCODE, ANN_DT from ASHAREPROSECUTION"""
           + """ where ANN_DT<='{}' and ANN_DT>='{}'""".format(date, date_p1y))
    print('Reading sql AShareProsecution')
    df_prosecution = pd.read_sql_query(sql, conn)
    df_prosecution.rename({'S_INFO_WINDCODE': 'sid'}, axis=1, inplace=True)
    df_prosecution = df_prosecution[df_prosecution.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    #sr_prosecution = 1.0 * df_prosecution.groupby("sid").apply(lambda x:(len(x)>3)) #哑变量型
    sr_prosecution = 1.0 * df_prosecution.groupby("sid").apply(lambda x:len(x)) #连续变量型
    
    # 业绩预告不准确度, 20191101会剔除761只股票
    sql = ("""SELECT S_INFO_WINDCODE, S_PROFITNOTICE_DATE, S_PROFITNOTICE_PERIOD, S_PROFITNOTICE_NETPROFITMIN from ASHAREPROFITNOTICE"""
          + """ where S_PROFITNOTICE_DATE<='{}' and S_PROFITNOTICE_PERIOD>='{}'""".format(date, date_p1y))
    print('Reading sql AShareProfitNotice')
    df_profit_notice = pd.read_sql(sql,conn)
    df_profit_notice.columns = [ele.lower() for ele in df_profit_notice.columns]
    df_profit_notice.rename({'s_info_windcode': 'sid', 's_profitnotice_period': 'report_period'}, axis=1, inplace=True)
    df_profit_notice.s_profitnotice_date = df_profit_notice.s_profitnotice_date.astype('i4')
    df_profit_notice.report_period = df_profit_notice.report_period.astype('i4')
    df_profit_notice = df_profit_notice[df_profit_notice.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_profit_notice.sort_values(['sid', 'report_period', 's_profitnotice_date'], inplace=True)
    # 财报
    sql = ("""select S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, NET_PROFIT_INCL_MIN_INT_INC from ASHAREINCOME"""
           + """ where STATEMENT_TYPE ='408001000' and ANN_DT<='{}' and REPORT_PERIOD>='{}'""".format(date, date_p1y))
    df_income = pd.read_sql_query(sql,conn)
    df_income.columns = [ele.lower() for ele in df_income.columns]
    df_income.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_income.ann_dt = df_income.ann_dt.astype('i4')
    df_income.report_period = df_income.report_period.astype('i4')
    df_income = df_income[df_income.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_income.sort_values(['sid', 'report_period', 'ann_dt'], inplace=True)
    # 处理
    sr_notice_lowerbound = df_profit_notice.set_index(['sid', 'report_period'])['s_profitnotice_netprofitmin']
    sr_profit = df_income.set_index(['sid', 'report_period'])['net_profit_incl_min_int_inc']
    #sr_unaccuracy_notice = 1.0 * (sr_profit - sr_notice_lowerbound).groupby('sid').apply(lambda x: np.any(x<0.0))
    sr_unaccuracy_notice = 1.0 * ((sr_notice_lowerbound - sr_profit)/sr_notice_lowerbound).groupby('sid').apply(lambda x: np.nanmean(x))#连续型变量
    
    sr_issuing_delay.name = 'issuing_delay'
    sr_illegality.name = 'illegality'
    sr_reginv.name = 'reginv'
    sr_prosecution.name = 'prosecution'
    sr_unaccuracy_notice.name = 'unaccuracy_notice'
    return sr_issuing_delay, sr_illegality, sr_reginv, sr_prosecution, sr_unaccuracy_notice
    
    
## 商誉，并购次数
def get_over_invest_info(date, conn):
    date_p1y = get_prev_n_trade_date(date, 240)
    date_p3y = get_prev_n_trade_date(date, 720)
    
    sql = ("""select S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, TOT_ASSETS, TOT_LIAB, GOODWILL from ASHAREBALANCESHEET"""
           + """ where STATEMENT_TYPE='408001000' and ANN_DT<='{}' and REPORT_PERIOD>='{}'""".format(date, date_p1y))
    df_balance = pd.read_sql_query(sql,conn)
    print('Reading sql AShareBalanceSheet')
    df_balance = pd.read_sql_query(sql, conn)
    df_balance.columns = [ele.lower() for ele in df_balance.columns]
    df_balance.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_balance.ann_dt = df_balance.ann_dt.astype('i4')
    df_balance.report_period = df_balance.report_period.astype('i4')
    df_balance = df_balance[df_balance.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))] # & (~np.isnan(df_balance.tot_assets))]
    df_balance.sort_values(['sid', 'report_period', 'ann_dt'], inplace=True)
    ## 处理
    df_balance2 = df_balance.groupby("sid").last()
    sr_goodwill_ratio = df_balance2['goodwill'] / (np.abs(df_balance2['tot_assets'] - df_balance2['tot_liab']) + 1.e4)
    
    # 并购次数, 这个比较慢
    sql="""select EVENT_ID, S_INFO_WINDCODE from MERGERPARTICIPANT where RELATIONSHIP=323001000"""  #先获取关联并购的参与方
    print('Reading sql MergerParticipant')
    df_merger_participant = pd.read_sql(sql,conn)
    df_merger_participant.columns = [ele.lower() for ele in df_merger_participant.columns]
    df_merger_participant.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_merger_participant = df_merger_participant[df_merger_participant.sid.apply(lambda x:x is not None and ((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_merger_participant = df_merger_participant.groupby(['event_id', 'sid'], as_index=True).last().reset_index()  # 去重
    # 并购
    sql = ("""select EVENT_ID, ANN_DATE, UPDATE_DATE, IS_MAJORASSETRESTRUCTURE from MERGEREVENT"""
           + """ where UPDATE_DATE>='{}' and UPDATE_DATE<='{}'""".format(date_p3y, date))
    print('Reading sql MergerEvent')
    df_merger_event = pd.read_sql(sql,conn)
    df_merger_event.columns = [ele.lower() for ele in df_merger_event.columns]
    df_merger_event.update_date = df_merger_event.update_date.astype('i4')
    df_merger_event = df_merger_event.merge(df_merger_participant, on='event_id', how='inner')  #合并，取inner
    df_merger_event = df_merger_event[df_merger_event.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_merger_event.sort_values(['sid', 'update_date'], inplace=True)
    # 处理数据
    sr_merger_times = 1.0 * df_merger_event.groupby("sid").apply(lambda x:len(x))
    
    sr_goodwill_ratio.name = 'goodwill_ratio'
    sr_merger_times.name = 'merger_times'
    return sr_goodwill_ratio, sr_merger_times

    
###### Part 2 ######
# 关联交易（关联进货销货，关联融资余额，关联并购，关联交易次数）
def get_related_trade_info(date, conn):
    ###输出为一个元组，包含4个sr
    date_p1y = get_prev_n_trade_date(date, 240)
    date_p3y = get_prev_n_trade_date(date, 720)
    code_map = get_code_map(conn)
    def func_ratio(a, b, eps=1.e4):
        return a / (np.abs(b) + eps)
#         return (a - b) / (np.abs(a) + np.abs(b) + eps) 
    
    # 获取财务指标
    sql=(f"""select S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, TOT_ASSETS, TOT_LIAB from ASHAREBALANCESHEET"""
         + f""" where STATEMENT_TYPE='408001000' and REPORT_PERIOD>='%s' and ANN_DT<='%s'""" % (date_p3y, date))
    print('Reading sql AShareBalanceSheet')
    df_balance = pd.read_sql(sql,conn)
    df_balance.columns = [ele.lower() for ele in df_balance.columns]
    df_balance.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_balance.ann_dt = df_balance.ann_dt.astype('i4')
    df_balance.report_period = df_balance.report_period.astype('i4')
    df_balance = df_balance[df_balance.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))] # & (~np.isnan(df_balance.tot_assets)) & (~np.isnan(df_balance.tot_liab))]
    df_balance.sort_values(['sid', 'report_period', 'ann_dt'], inplace=True)
    ## 处理数据
    df_balance['net_assets'] = df_balance.tot_assets - df_balance.tot_liab
    df_balance_p1y = df_balance[df_balance.report_period >= date_p1y]
    sr_tot_assets_p3y = df_balance.groupby('sid', as_index=True)['tot_assets'].mean()  # 3年总资产平均值
    sr_net_assets = df_balance_p1y.groupby('sid', as_index=True)['net_assets'].last()  # 近期的净资产
    
    # 关联进货销货
    sql = ("""select S_INFO_WINDCODE, ANN_DT, S_RELATEDTRADE_TRADETYPE, CRNCY_CODE, S_RELATEDTRADE_AMOUNT from ASHARERALATEDTRADE"""
           + f""" where ANN_DT>='%s' and ANN_DT<='%s'""") % (date_p1y, date)
    print('Reading sql AShareRalatedTrade')
    df_related_trade = pd.read_sql(sql,conn)
    df_related_trade.columns = [ele.lower() for ele in df_related_trade.columns]
    df_related_trade.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_related_trade.ann_dt = df_related_trade.ann_dt.astype('i4')
    df_related_trade['s_relatedtrade_amount'] = df_related_trade['s_relatedtrade_amount'].replace([None, np.nan], '0').astype('f8')  # wind给的type是字符串
    df_related_trade = df_related_trade[df_related_trade.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_related_trade.sort_values(['sid', 'ann_dt'], inplace=True)
    #处理数据，通过对照公告，s_relatedtrade_amount单位为元
    type_sell_buy = ['出售', '向关联方采购产品和接受劳务', '向关联方销售产品和提供劳务', 
                     '接受劳务', '接受服务', '提供加工', '提供劳务', '提供服务', '购买', '购买商品', '购销', '购销,提供服务', '购销商品', 
                     '采购', '采购货物', '采购货物,接受劳务', '销售', '销售,提供', '销售,提供劳务', '销售商品', '销售货物']
    df_related_sellbuy = df_related_trade[df_related_trade.s_relatedtrade_tradetype.isin(type_sell_buy)
                                          & df_related_trade.crncy_code.apply(lambda x: x in ['CNY', None])]
    sr_related_sellbuy_amount = df_related_sellbuy.groupby('sid', as_index=True)['s_relatedtrade_amount'].sum()
    sr_related_sellbuy_ratio = func_ratio(sr_related_sellbuy_amount, sr_net_assets)   # 关联进货销货
    
    # 关联融资余额, 也有报告期
    sql = ("""select S_INFO_COMPCODE, ANN_DT, REPORT_PERIOD, ASSOCIATED_NAME, ASSOCIATED_FUNDING_BALANCE, CRNCY_CODE from ASHARERELATEDCLAIMSDEBTS """
           + """ where REPORT_PERIOD>='%s' and ANN_DT<='%s'""" % (date_p1y, date))
    print('Reading sql AShareRelatedclaimsdebts')
    df_related_debts = pd.read_sql(sql,conn)
    df_related_debts.columns = [ele.lower() for ele in df_related_debts.columns]
    df_related_debts.ann_dt = df_related_debts.ann_dt.astype('i4')
    df_related_debts.report_period = df_related_debts.report_period.astype('i4')
    df_related_debts['sid'] = [code_map.get(ele, '0') for ele in df_related_debts.s_info_compcode]
    df_related_debts = df_related_debts[df_related_debts.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_related_debts.sort_values(['sid', 'report_period'], inplace=True)
    ## 处理数据， 通过对照20190630报告期，000005.SZ和	603888.SH，可以推断余额是万元单位
    df_related_debts = df_related_debts[df_related_debts.crncy_code.apply(lambda x: x in ['CNY', None])]
    sr_related_debts_balance = 10000.0 * df_related_debts.groupby(['sid', 'report_period'], as_index=True)['associated_funding_balance'].sum().groupby('sid').last()
    sr_related_debts_ratio = func_ratio(sr_related_debts_balance, sr_net_assets)  # 关联融资余额
    
    # 关联并购
    sql="""select EVENT_ID, S_INFO_WINDCODE from MERGERPARTICIPANT where RELATIONSHIP=323001000"""  # 获取关联并购的参与方
    print('Reading sql MergerParticipant')
    df_merger_participant = pd.read_sql(sql,conn)
    df_merger_participant.columns = [ele.lower() for ele in df_merger_participant.columns]
    df_merger_participant.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_merger_participant = df_merger_participant[df_merger_participant.sid.apply(lambda x:x is not None and ((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_merger_participant = df_merger_participant.groupby(['event_id', 'sid'], as_index=True).last().reset_index()
    # 获取关联并购
    sql = ("""select EVENT_ID, ANN_DATE, UPDATE_DATE, IS_RELATED_PARTY_TRANSAC, PROGRESS_CODE, TRADE_VALUE, CRNCY_CODE from MERGEREVENT"""
           + """ where IS_RELATED_PARTY_TRANSAC=1 and UPDATE_DATE>='%s' and UPDATE_DATE<='%s'""" % (date_p3y, date))
    print('Reading sql MergerEvent')
    df_related_merger = pd.read_sql(sql,conn)
    df_related_merger.columns = [ele.lower() for ele in df_related_merger.columns]
#     df_related_merger.ann_date = df_related_merger.ann_date.astype('i4'), not use
    df_related_merger.update_date = df_related_merger.update_date.astype('i4')
    df_related_merger = df_related_merger.merge(df_merger_participant, on='event_id',how='inner')  #合并，取inner
    df_related_merger = df_related_merger[df_related_merger.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_related_merger.sort_values(['sid', 'update_date'], inplace=True)
    # 处理数据
    df_related_merger = df_related_merger[df_related_merger.crncy_code.apply(lambda x: x in ['CNY', None])
                                          & df_related_merger.progress_code.apply(lambda x: x in [324004000, 324004001, 324004002, 324004003])]
    df_related_merger_amount = 10000.0 * df_related_merger.groupby('sid', as_index=True)['trade_value'].sum()
    df_related_merger_ratio = func_ratio(df_related_merger_amount, sr_tot_assets_p3y)  # 关联并购
    
    #  关联交易次数
    df_related_trade_num = df_related_trade.groupby('sid', as_index=True).apply(len)
    
    # 输出结果
    sr_related_sellbuy_ratio.name = 'related_sellbuy_ratio'
    sr_related_debts_ratio.name = 'related_debts_ratio'
    df_related_merger_ratio.name = 'related_merger_ratio'
    df_related_trade_num.name = 'related_trade_num'
    return sr_related_sellbuy_ratio, sr_related_debts_ratio, df_related_merger_ratio, df_related_trade_num
    
    
## 分析师预期（评级周下调，目标价周下调，预期eps周下调）
def get_analyst_expect_info(date, conn):
    date_p5d = get_prev_n_trade_date(date, 5)
    date_p1m = get_prev_n_trade_date(date, 20)
    date_p3m = get_prev_n_trade_date(date, 60)  #先获取近3个月的数据
    
    # 获取评级周下调, 分值越低评级越高，周期有三个30天、90天河180天先不管
    sql=("""select S_INFO_WINDCODE, RATING_DT, S_WRATING_CYCLE, S_WRATING_AVG, S_EST_PRICE from ASHARESTOCKRATINGCONSUS"""
         + """ where RATING_DT>='%s' and RATING_DT<='%s'""" % (date_p3m, date))
    print('Reading sql AShareStockRatingConsus')
    df_stock_rating=pd.read_sql(sql,conn)
    df_stock_rating.columns = [ele.lower() for ele in df_stock_rating.columns]
    df_stock_rating.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_stock_rating.rating_dt = df_stock_rating.rating_dt.astype('i4')
    df_stock_rating = df_stock_rating[df_stock_rating.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_stock_rating.sort_values(['sid', 'rating_dt', 's_wrating_cycle'], inplace=True)
    # 评级周下调
    df_stock_rating_early = df_stock_rating[df_stock_rating.rating_dt <= date_p5d]
    df_stock_rating_p5d = df_stock_rating[df_stock_rating.rating_dt > date_p5d]
    df_stock_rating_early2 = df_stock_rating_early[~np.isnan(df_stock_rating_early.s_wrating_avg)]
    df_stock_rating_p5d2 = df_stock_rating_p5d[~np.isnan(df_stock_rating_p5d.s_wrating_avg)]
    sr_rating_early = df_stock_rating_early2.groupby('sid', as_index=True)['s_wrating_avg'].last()
    sr_rating_last = df_stock_rating_p5d2.groupby('sid', as_index=True)['s_wrating_avg'].last()
    sr_rating_delta = sr_rating_last - sr_rating_early  # 大于0表示评级下降
    
    # 目标价周下调
    df_stock_rating_early3 = df_stock_rating_early[~np.isnan(df_stock_rating_early.s_est_price)]
    df_stock_rating_p5d3 = df_stock_rating_p5d[~np.isnan(df_stock_rating_p5d.s_est_price)]
    sr_est_price_early = df_stock_rating_early3.groupby('sid', as_index=True)['s_est_price'].last()
    sr_est_price_last = df_stock_rating_p5d3.groupby('sid', as_index=True)['s_est_price'].last()
    sr_est_price_delta = sr_est_price_last - sr_est_price_early  # 目标价周下调
    
    # 预期eps周下调， 这张表每天更新，跑起来需要1~2min,所以用past 1month
    sql = ("""select S_INFO_WINDCODE, EST_DT, ROLLING_TYPE, EST_EPS from ASHARECONSENSUSROLLINGDATA"""
           + """ where ROLLING_TYPE='FY1' and EST_DT>='%s' and EST_DT<='%s'""" % (date_p1m, date))
    print('Reading sql AShareConsensusRollingData')
    df_con_eps = pd.read_sql(sql, conn)
    df_con_eps.columns = [ele.lower() for ele in df_con_eps.columns]
    df_con_eps.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_con_eps.est_dt = df_con_eps.est_dt.astype('i4')
    df_con_eps = df_con_eps[df_con_eps.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_con_eps.sort_values(['sid', 'est_dt'], inplace=True)
    # 处理数据
    df_con_eps_early = df_con_eps[df_con_eps.est_dt <= date_p5d]
    df_con_eps_early = df_con_eps_early[~np.isnan(df_con_eps_early.est_eps)]
    df_con_eps_p5d = df_con_eps[df_con_eps.est_dt > date_p5d]
    df_con_eps_p5d = df_con_eps_p5d[~np.isnan(df_con_eps_p5d.est_eps)]
    sr_eps_early = df_con_eps_early.groupby('sid', as_index=True)['est_eps'].last()
    sr_eps_last = df_con_eps_p5d.groupby('sid', as_index=True)['est_eps'].last()
    sr_eps_delta = sr_eps_last - sr_eps_early  # 预期eps周下调
    
    # 输出结果
    sr_rating_delta.name = 'rating_delta'
    sr_est_price_delta.name = 'est_price_delta'
    sr_eps_delta.name = 'eps_delta'
    return sr_rating_delta, sr_est_price_delta, sr_eps_delta
    
    
## 减持（大股东减持、高管减持）
def get_holder_trade_info(date, conn):
    date_p3m = get_prev_n_trade_date(date, 60)  #先获取近3个月的数据
    
    # 先获取总股本，单位为万股
    sql = """select S_INFO_WINDCODE, CHANGE_DT, TOT_SHR from ASHARECAPITALIZATION where CHANGE_DT<={}""".format(date)
    print('Reading sql AShareCapitalization')
    df_tot_share = pd.read_sql_query(sql, conn)
    df_tot_share.rename({'S_INFO_WINDCODE': 'sid', 'CHANGE_DT': 'change_dt', 'TOT_SHR': 'tot_shr'}, axis=1, inplace=True)
    df_tot_share['change_dt'] = df_tot_share['change_dt'].values.astype('i4')
    df_tot_share = df_tot_share[df_tot_share.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_tot_share.sort_values(by=['sid', 'change_dt'], inplace=True)
    sr_tot_share = 10000.0 * df_tot_share.groupby('sid')['tot_shr'].last()
    
    # 大股东减持，里面的单位为股
    sql = ("""select S_INFO_WINDCODE, ANN_DT, HOLDER_TYPE, TRANSACT_TYPE, TRANSACT_QUANTITY, NEW_HOLD_TOT from ASHAREMJRHOLDERTRADE"""
           + """ where TRANSACT_TYPE='减持' and ANN_DT>='%s' and ANN_DT<='%s'""" % (date_p3m, date)) # 因为有起始日期和截至日期，所以我们还是以ann_dt为准
    print('Reading sql AShareMjrHolderTrade')
    df_holder_trade = pd.read_sql(sql, conn)
    df_holder_trade.columns = [ele.lower() for ele in df_holder_trade.columns]
    df_holder_trade.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_holder_trade.ann_dt = df_holder_trade.ann_dt.astype('i4')
    df_holder_trade = df_holder_trade[df_holder_trade.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_holder_trade.sort_values(['sid', 'ann_dt'], inplace=True)
    # 大股东减持
    df_holder_trade = df_holder_trade.merge(sr_tot_share, on='sid', how='left')
    df_holder_trade['new_hold_tot'] = df_holder_trade['new_hold_tot'].fillna(0.0)
    df_holder_trade['trade_ratio'] = 1.0 * df_holder_trade.transact_quantity / df_holder_trade.tot_shr
    df_holder_trade2 = df_holder_trade[df_holder_trade.new_hold_tot + df_holder_trade.transact_quantity >= 0.05*df_holder_trade.tot_shr]
    sr_holder_sell = df_holder_trade2.groupby('sid')['trade_ratio'].sum()
    
    # 高管减持
    df_holder_trade2 = df_holder_trade[df_holder_trade.holder_type=='3']
    sr_manager_sell = df_holder_trade2.groupby('sid')['trade_ratio'].sum()
    
    # 返回结果
    sr_holder_sell.name = 'holder_sell'
    sr_manager_sell.name = 'manager_sell'
    return sr_holder_sell, sr_manager_sell  #注意，如果得到总股本为np.nan，即使有减持，也会返回减持结果为0，我们暂且忽略
    
    
## 重组失败，限售股解禁，债券违约、问询函
def get_risk_event_info(date, conn):
    date_p5d = get_prev_n_trade_date(date, 5)
    date_p1m = get_prev_n_trade_date(date, 20)
    date_p6m = get_prev_n_trade_date(date, 120)
    date_p2y = get_prev_n_trade_date(date, 480)
    date_f15d = get_next_n_trade_date(date, 10+5)  # 未来两周，因为是每周更新，保险起见多加个5天
    
    # 重组失败
    sql="""select EVENT_ID, S_INFO_WINDCODE from MERGERPARTICIPANT where RELATIONSHIP=323001000"""  # 获取关联并购的参与方
    print('Reading sql MergerParticipant')
    df_merger_participant = pd.read_sql(sql,conn)
    df_merger_participant.columns = [ele.lower() for ele in df_merger_participant.columns]
    df_merger_participant.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_merger_participant = df_merger_participant[df_merger_participant.sid.apply(lambda x:x is not None and ((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_merger_participant = df_merger_participant.groupby(['event_id', 'sid'], as_index=True).last().reset_index()
    # 获取关联并购
    sql = ("""select EVENT_ID, ANN_DATE, UPDATE_DATE, PROGRESS_CODE from MERGEREVENT"""
           + """ where IS_MAJORASSETRESTRUCTURE=1 and UPDATE_DATE>='%s' and UPDATE_DATE<='%s'""" % (date_p6m, date))
    print('Reading sql MergerEvent')
    df_merger_event = pd.read_sql(sql,conn)
    df_merger_event.columns = [ele.lower() for ele in df_merger_event.columns]
    df_merger_event.update_date = df_merger_event.update_date.astype('i4')
    df_merger_event = df_merger_event.merge(df_merger_participant, on='event_id', how='inner')  #合并，取inner
    df_merger_event = df_merger_event[df_merger_event.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_merger_event.sort_values(['sid', 'update_date'], inplace=True)
    # 处理数据
    df_merger_event = df_merger_event[df_merger_event.progress_code.apply(lambda x: x in  [324005000, 324005001, 324005002, 324005003, 324005004, 324005005])]
    sr_merger_failure = 1.0 * (df_merger_event.groupby('sid').apply(len) > 0) # 重组失败
    
    # 限售股解禁, 2010年以来存在4209/3593632约0.117%的情况ANN_DT为Null，我们暂且忽略
    sql=("""select S_INFO_WINDCODE, ANN_DT, S_INFO_LISTDATE, S_SHARE_LST, S_SHARE_RATIO from ASHARECOMPRESTRICTED"""
         + """ where ANN_DT<='%s' and S_INFO_LISTDATE>'%s' and S_INFO_LISTDATE<='%s'""" % (date, date_p5d, date_f15d))  #(过去五天，未来15天]
    print('Reading sql AShareCompRestricted')
    df_restricted_list = pd.read_sql(sql,conn)
    df_restricted_list.columns = [ele.lower() for ele in df_restricted_list.columns]
    df_restricted_list.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_restricted_list.ann_dt = df_restricted_list.ann_dt.astype('i4')
    df_restricted_list = df_restricted_list[df_restricted_list.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_restricted_list.sort_values(['sid', 'ann_dt'], inplace=True)
    # 计算
    sr_restricted_ratio = 0.01 * df_restricted_list.groupby('sid')['s_share_ratio'].sum() # 限售股解禁
    
    # 债券违约
    sql=("""select S_INFO_WINDCODE, S_EVENT_ANNCEDATE, S_EVENT_CONTENT from ASHAREMAJOREVENT""" 
         + """ where S_EVENT_ANNCEDATE>='%s' and S_EVENT_ANNCEDATE<='%s'""" % (date_p2y, date))
    print('Reading sql AShareMajorEvent')
    df_major_event = pd.read_sql(sql, conn)
    df_major_event.columns = [ele.lower() for ele in df_major_event.columns]
    df_major_event.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_major_event.s_event_anncedate = df_major_event.s_event_anncedate.astype('i4')
    df_major_event = df_major_event[df_major_event.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_major_event.sort_values(['sid', 's_event_anncedate'], inplace=True)
    # 处理
    bond_default_list = ['担保违约','担保人代偿','兑付风险警示','未按时兑付本金','未按时兑付利息','未按时兑付本息',
                         '未按时兑付回售款', '未按时兑付回售款和利息', '提前到期未兑付','技术性违约']
    def func_bond_default(x):
        if x is None:
            return False
        for ele in bond_default_list:
            if ele in x:
                return True
        return ('债券' in x and '违约' in x)
    df_major_event2 = df_major_event[df_major_event.s_event_content.apply(lambda x: func_bond_default(x))]
    sr_bond_default = 1.0 * (df_major_event2.groupby('sid').apply(len) > 0)
    
    #问询函
    df_major_event3 = df_major_event[df_major_event.s_event_anncedate >= date_p1m]  # 最近一个月
    df_major_event3 = df_major_event3[df_major_event3.s_event_content.apply(lambda x: x is not None and '问询函' in x)]
    sr_enquiry_letter = 1.0 * (df_major_event3.groupby('sid').apply(len) > 0)
    
    sr_merger_failure.name = 'merger_failure'
    sr_restricted_ratio.name = 'restricted_ratio'
    sr_bond_default.name = 'bond_default'
    sr_enquiry_letter.name = 'enquiry_letter'
    return sr_merger_failure, sr_restricted_ratio, sr_bond_default, sr_enquiry_letter
    
    
## 财务报告（业绩实亏、业绩预亏、业绩预告大幅度下滑、审计意见，政府补助）
def get_finance_report_info(date, conn):
    date_p1y = get_prev_n_trade_date(date, 240)
    code_map = get_code_map(conn)
    
    # 业绩实际亏损
    sql=(f"""select S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, NET_PROFIT_INCL_MIN_INT_INC from ASHAREINCOME"""
         + f""" where STATEMENT_TYPE='408001000' and REPORT_PERIOD>='%s' and ANN_DT<='%s'""" % (date_p1y, date))
    print('Reading sql AShareIncome')
    df_income = pd.read_sql(sql,conn)
    df_income.columns = [ele.lower() for ele in df_income.columns]
    df_income.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_income.ann_dt = df_income.ann_dt.astype('i4')
    df_income.report_period = df_income.report_period.astype('i4')
    df_income = df_income[df_income.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))] # & (~np.isnan(df_income.net_profit_incl_min_int_inc))]
    df_income.sort_values(['sid', 'report_period', 'ann_dt'], inplace=True)
    # 处理定期财务报告数据
    sr_income_period = df_income.groupby('sid')['report_period'].last()
    sr_profit = df_income.groupby('sid')['net_profit_incl_min_int_inc'].last()
    sr_actual_deficit = 1.0 * (sr_profit < 0.0)
    
    # 业绩预亏(对比了一下，上年同期归母净利润等三项单位都是万元)，需要观察预报是否领先于财报，领先财报的sid才算快报
    sql=(f"""select S_INFO_WINDCODE, S_PROFITNOTICE_DATE, S_PROFITNOTICE_PERIOD, S_PROFITNOTICE_NETPROFITMIN, S_PROFITNOTICE_NETPROFITMAX, S_PROFITNOTICE_NET_PARENT_FIRM from ASHAREPROFITNOTICE"""
         + f""" where S_PROFITNOTICE_PERIOD>='%s' and S_PROFITNOTICE_DATE<='%s'""" % (date_p1y, date))
    print('Reading sql AShareProfitNotice')
    df_profit_notice = pd.read_sql(sql,conn)
    df_profit_notice.columns = [ele.lower() for ele in df_profit_notice.columns]
    df_profit_notice.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_profit_notice.s_profitnotice_date = df_profit_notice.s_profitnotice_date.astype('i4')
    df_profit_notice.s_profitnotice_period = df_profit_notice.s_profitnotice_period.astype('i4')
    df_profit_notice = df_profit_notice[df_profit_notice.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_profit_notice.sort_values(['sid', 's_profitnotice_period', 's_profitnotice_date'], inplace=True)
    # 处理
    df_profit_notice = df_profit_notice.groupby('sid', as_index=True).last()
    df_profit_notice = df_profit_notice.merge(sr_income_period, on='sid', how='left')
    df_profit_notice.report_period = df_profit_notice.report_period.replace(np.nan, 0).astype('i4')
    df_profit_notice = df_profit_notice[df_profit_notice.s_profitnotice_period > df_profit_notice.report_period] # 预告要在财务报告之后
    sr_notice_deficit = 1.0 * (df_profit_notice.s_profitnotice_netprofitmin + df_profit_notice.s_profitnotice_netprofitmax < 0.0)  # 业绩预亏
    
    # 业绩大幅下滑
    def func_collapse(x):
        profit_notice_mean = 0.5 * (x.s_profitnotice_netprofitmin + x.s_profitnotice_netprofitmax)
        profit_pre = 1.0 * x.s_profitnotice_net_parent_firm
        if profit_pre < 0:
            return profit_notice_mean < 1.5 * profit_pre
        elif profit_pre >= 0:
            return profit_notice_mean < 0.5 * profit_pre
        else:   # np.nan
            return False
    sr_notice_collapse = 1.0 * pd.Series([func_collapse(ele) for _, ele in df_profit_notice.iterrows()], index=df_profit_notice.index)
    
    # 审计意见
    sql = ("""select S_INFO_WINDCODE, ANN_DT, REPORT_PERIOD, S_STMNOTE_AUDIT_CATEGORY from ASHAREAUDITOPINION"""
           + f""" where REPORT_PERIOD>='%s' and ANN_DT<='%s'""" % (date_p1y, date))
    print('Reading sql AShareAuditOpinion')
    df_audit_opinion = pd.read_sql(sql,conn)
    df_audit_opinion.columns = [ele.lower() for ele in df_audit_opinion.columns]
    df_audit_opinion.rename({'s_info_windcode': 'sid'}, axis=1, inplace=True)
    df_audit_opinion.ann_dt = df_audit_opinion.ann_dt.astype('i4')
    df_audit_opinion.report_period = df_audit_opinion.report_period.astype('i4')
    df_audit_opinion = df_audit_opinion[df_audit_opinion.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_audit_opinion.sort_values(['sid', 'report_period', 'ann_dt'], inplace=True)
    # 处理数据
    df_audit_opinion = df_audit_opinion.groupby('sid', as_index=True).last()
    sr_audit_opinion = 1.0 * (df_audit_opinion.s_stmnote_audit_category != 405001000)    
    
    # 政府补助，所有项加在一起，表中单位为元，和公告也对了一下
    sql=(f"""select S_INFO_COMPCODE, ANN_DATE, REPORT_PERIOD, ITEM_NAME, AMOUNT_CURRENT_ISSUE from ASHAREGOVERNMENTGRANTS"""
         + f""" where REPORT_PERIOD>='%s' and ANN_DATE<='%s'""" % (date_p1y, date))
    print('Reading sql AShareGovernmentgrants')
    df_gov_grants= pd.read_sql(sql,conn)
    df_gov_grants.columns = [ele.lower() for ele in df_gov_grants.columns]
    df_gov_grants.ann_date = df_gov_grants.ann_date.astype('i4')
    df_gov_grants.report_period = df_gov_grants.report_period.astype('i4')
    df_gov_grants['sid'] = [code_map.get(ele, '0') for ele in df_gov_grants.s_info_compcode]
    df_gov_grants = df_gov_grants[df_gov_grants.sid.apply(lambda x:((x[::8] in ['0Z','3Z','6H']) & (len(x)==9)))]
    df_gov_grants.sort_values(['sid', 'report_period', 'ann_date'], inplace=True)
    # 处理数据
    sr_grants_amount = df_gov_grants.groupby(['sid', 'report_period'], as_index=True)['amount_current_issue'].sum().groupby('sid').last()
    sr_grants_ratio = sr_grants_amount / (np.abs(sr_profit) + 1.e4)  # 先这么定义吧
    
    sr_actual_deficit.name = 'actual_deficit'
    sr_notice_deficit.name = 'notice_deficit'
    sr_notice_collapse.name = 'notice_collapse'
    sr_audit_opinion.name = 'audit_opinion'
    sr_grants_ratio.name = 'grants_ratio'
    return sr_actual_deficit, sr_notice_deficit, sr_notice_collapse, sr_audit_opinion, sr_grants_ratio
