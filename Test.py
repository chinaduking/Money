##### 以下代码为标准格式 #####
import jqdata
from jqlib.alpha191 import *
import scipy.optimize as sco
from kuanke.wizard import *

from jqdata import *
from functools import reduce
import talib as tl
import numpy as np
import pandas as pd
import matplotlib  # 注意这个也要import一次
import matplotlib as plt
import datetime


# 初始化函数，设定要操作的股票、基准等等
def initialize(context):
    # 参数设置
    set_params()
    # 设置回测的参数
    set_backtest()
    # 每周运行一次
    run_weekly(WeeklyTactics, 1, 'before_open')


# 按照回测频率进行调用
def handle_data(context, data):
    StockOperate(context)


#################################USER################################
# 参数设置
def set_params():
    # 需要购买股票的股票池
    g.buy_stock = []


# 回测设置
def set_backtest():
    # 使用真是数据进行回测
    set_option('use_real_price', True)
    # 设置持仓股票数量
    g.count = 5
    # 设定KDJ指标初始值
    g.K1 = 50
    g.D1 = 50


# 每周执行的策略
def WeeklyTactics(context):
    # alp = Alphas191_Rank(context)
    alp = FAP_CMA_Rank(context)
    # alp = kdj_filter(alp)   #KDJ  选股策略
    # alp = check_PB_stocks(context)

    if len(list(alp.index)) > g.count:
        g.buy_stock = list(alp.index)[0:g.count]
    elif len(alp) != 0:
        g.buy_stock = list(alp.index)[0:len(alp)]

    print('g.buy_stock:', g.buy_stock)


# 回测的操作
def StockOperate(context):
    day = context.current_dt.day
    if day > 24:
        for stock in g.buy_stock:
            order_target_value(stock, 0)
        return
    reset_position(context)  # 轮换清仓
    # 指数止损
    if conduct_dapan_stoploss(context, '000001.XSHG', 4, -0.05):
        return
    port_weight(context)


##############################################################

# del 股票过滤
def del_st_paused(code):
    current_data = get_current_data(code)
    code = [s for s in code if not current_data[s].is_st]  # 过滤ST
    code = [s for s in code if not current_data[s].paused]  # 过滤停牌
    code = [s for s in code if not current_data[s].day_open >= current_data[s].high_limit]  # 过滤涨停
    code = [s for s in code if not current_data[s].day_open <= current_data[s].low_limit]  # 过滤跌停
    # for s in code:
    #     print(current_data[s].paused,current_data[s].is_st)
    return code


# Alphas191 策略
def Alphas191_Rank(context):
    # current_data = context.current_dt.strftime("%Y-%m-%d")  #当前回测时间
    code = get_index_stocks('000001.XSHG')  # 股票代码
    code = del_st_paused(code)  # del ST %% paused

    d1 = context.current_dt.date()
    d2 = d1 - datetime.timedelta(days=1)
    d3 = d2 - datetime.timedelta(days=1)

    # alpha_005 因子    B+
    alp = alpha_005(code, d2)
    alp = alp.sort(ascending=False)

    # # alpha_002 因子
    # alp = alpha_002(code, d2)
    # alp = alp.sort_values(ascending = False)
    return alp


# 多因子模型+资产组合优化
def FAP_CMA_Rank(context):
    # CMV，FAP因子越小收益越大,分值越大
    effective_factors = {'FAP': False, 'CMV': False}
    # FAP:固定资产比率,越小越好
    # CMV:流通市值,越小越好
    # 获取所有指股票
    fdf = get_factors('000001.XSHG', context)

    # 数据归一化
    fdf['FAP'] = (fdf['FAP'] - fdf['FAP'].min()) / (fdf['FAP'].max() - fdf['FAP'].min())
    fdf['CMV'] = (fdf['CMV'] - fdf['CMV'].min()) / (fdf['CMV'].max() - fdf['CMV'].min())
    fdf_norm = fdf

    fdf_norm['CMV'] = 1 - fdf_norm['CMV']
    fdf_norm['FAP'] = 1 - fdf_norm['FAP']

    avr_fa = fdf_norm['FAP'].mean()
    avr_cm = fdf_norm['CMV'].mean()

    K4 = 4 / avr_fa  # SS
    K5 = 5 / avr_cm  # SSS

    # 权重 矩阵相乘
    rank = (fdf_norm * np.array([K4, K5])).T.sum().order(ascending=False)
    return rank


## 股票筛选并排序    低PB价值投资策略
def check_PB_stocks(context):
    code = get_index_stocks('000001.XSHG')  # 股票代码
    code = del_st_paused(code)  # del ST %% paused
    # 财务筛选
    out_lists = financial_statements_filter(context, code)
    # out_lists = code

    # 经过以上筛选后的股票根据市净率大小按升序
    df_check_out_lists = get_fundamentals(query(
        valuation.code, valuation.pb_ratio
    ).filter(
        # 这里不能使用 in 操作, 要使用in_()函数
        valuation.code.in_(out_lists)
    ).order_by(
        # 按市净率升序排列，排序准则：desc-降序、asc-升序
        valuation.pb_ratio.asc()
    ).limit(
        # 最多返回10个
        10
        # 前一个交易日的日期
    ), date=context.previous_date)

    # 筛选结果加入g.check_out_lists中
    check_out_lists = df_check_out_lists['code']
    # 排序
    input_dict = get_check_stocks_sort_input_dict()
    check_out = check_stocks_sort(context, check_out_lists, input_dict, 'desc')

    return check_out


################################################################
# 获取选股排序的 input_dict
def get_check_stocks_sort_input_dict():
    # desc-降序、asc-升序
    input_dict = {
        indicator.roe: ('desc', 0.7),  # 净资产收益率ROE，从大到小，权重0.7
        valuation.pb_ratio: ('asc', 0.05),  # 市净率，从小到大，权重0.05
        indicator.inc_net_profit_year_on_year: ('desc', 0.2),  # 净利润同比增长率，从大到小，权重0.2
        valuation.market_cap: ('desc', 0.05),  # 总市值，从大到小，权重0.05
    }
    # 返回结果
    return input_dict


## 排序
def check_stocks_sort(context, security_list, input_dict, ascending='desc'):
    if (len(security_list) == 0) or (len(input_dict) == 0):
        return security_list
    else:
        # 生成 key 的 list
        idk = list(input_dict.keys())
        # 生成矩阵
        a = pd.DataFrame()
        for i in idk:
            b = get_sort_dataframe(security_list, i, input_dict[i])
            a = pd.concat([a, b], axis=1)
        # 生成 score 列
        a['score'] = a.sum(1, False)
        # 根据 score 排序
        if ascending == 'asc':  # 升序
            a = a.sort(['score'], ascending=True)
        elif ascending == 'desc':  # 降序
            a = a.sort(['score'], ascending=False)
        # 返回结果
        return a


## 财务指标筛选函数
def financial_statements_filter(context, security_list):
    # 流通股本小于250000万股
    security_list = financial_data_filter_xiaoyu(security_list, valuation.circulating_cap, 250000)
    # 市净率小于0.85
    security_list = financial_data_filter_xiaoyu(security_list, valuation.pb_ratio, 0.85)
    # 营业收入同比增长率(%)：检验上市公司去年一年挣钱能力是否提高的标准
    security_list = financial_data_filter_dayu(security_list, indicator.inc_revenue_year_on_year, 0)
    # 净利润同比增长率：（当期的净利润-上月（上年）当期的净利润）/上月（上年）当期的净利润=净利润同比增长率
    security_list = financial_data_filter_dayu(security_list, indicator.inc_net_profit_year_on_year, 0)
    # 净资产收益率ROE：归属于母公司股东的净利润*2/（期初归属于母公司股东的净资产+期末归属于母公司股东的净资产）
    security_list = financial_data_filter_dayu(security_list, indicator.roe, 0)
    # 返回列表
    return security_list


# 获取金融数据
def get_factors(code, context):
    factors = ['FAP', 'CMV']
    d1 = context.current_dt.date()
    d2 = d1 - datetime.timedelta(days=1)
    stock_set = get_index_stocks(code, d2)
    stock_set = del_st_paused(list(stock_set))  # del ST %% paused
    q = query(
        valuation.code,
        balance.fixed_assets / balance.total_assets,
        valuation.circulating_market_cap
    ).filter(
        valuation.code.in_(stock_set)
    )
    fdf = get_fundamentals(q)
    fdf.index = fdf['code']
    fdf.columns = ['code'] + factors
    return fdf.iloc[:, -2:]


# 轮换选股后清除新股票池外的持仓  #卖出不在股票池里的股票
def reset_position(context):
    if context.portfolio.positions.keys() != []:
        for stock in context.portfolio.positions.keys():
            if stock not in g.buy_stock:
                order_target_value(stock, 0)


# 指数止损
def conduct_dapan_stoploss(context, security_code, days, bench):
    hist1 = attribute_history(security_code, days + 1, '1d', 'close', df=False)
    security_returns = (hist1['close'][-1] - hist1['close'][0]) / hist1['close'][0]
    if security_returns < bench:
        for stock in g.buy_stock:
            order_target_value(stock, 0)
            log.info("Sell %s for dapan nday stoploss" % stock)
        return True
    else:
        return False


def statistics(weights):
    weights = np.array(weights)
    port_returns = np.sum(g.returns.mean() * weights) * 252
    port_variance = np.sqrt(np.dot(weights.T, np.dot(g.returns.cov() * 252, weights)))
    return np.array([port_returns, port_variance, port_returns / port_variance])


# 最小化夏普指数的负值
def min_sharpe(weights):
    return -statistics(weights)[2]


def min_variance(weights):
    return statistics(weights)[1]


def port_weight(context):
    noa = len(g.buy_stock)
    if noa == 0:
        return 0

    df = history(400, '1d', 'close', g.buy_stock, df=True)
    g.returns = np.log(df / df.shift(1))

    # 约束是所有参数(权重)的总和为1。这可以用minimize函数的约定表达如下
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # 我们还将参数值(权重)限制在0和1之间。这些值以多个元组组成的一个元组形式提供给最小化函数
    bnds = tuple((0, 1) for x in range(noa))

    # 优化函数调用中忽略的唯一输入是起始参数列表(对权重的初始猜测)。我们简单的使用平均分布。
    optv = sco.minimize(min_variance, noa * [1. / noa, ], method='SLSQP', bounds=bnds, constraints=cons)
    print(optv['x'].round(3))
    # opts = sco.minimize(min_sharpe, noa*[1./noa,], method = 'SLSQP', bounds = bnds, constraints = cons)
    # print opts['x'].round(3)
    port_value = context.portfolio.portfolio_value
    for stock in g.buy_stock:
        buyValue = (optv['x'][g.buy_stock.index(stock)] * port_value)
        order_target_value(stock, buyValue)


##################################
# 同花顺和通达信等软件中的SMA
def SMA_CN(close, timeperiod):
    close = np.nan_to_num(close)
    return reduce(lambda x, y: ((timeperiod - 1) * x + y) / timeperiod, close)


# 同花顺和通达信等软件中的KDJ
def KDJ_CN(high, low, close, fastk_period, slowk_period, fastd_period):
    kValue, dValue = tl.STOCHF(high, low, close, fastk_period, fastd_period=1, fastd_matype=0)

    kValue = np.array(map(lambda x: SMA_CN(kValue[:x], slowk_period), range(1, len(kValue) + 1)))
    dValue = np.array(map(lambda x: SMA_CN(kValue[:x], fastd_period), range(1, len(kValue) + 1)))

    jValue = 3 * kValue - 2 * dValue

    func = lambda arr: np.array([0 if x < 0 else (100 if x > 100 else x) for x in arr])

    kValue = func(kValue)
    dValue = func(dValue)
    jValue = func(jValue)
    return kValue, dValue, jValue


# 同花顺和通达信等软件中的RSI
def RSI_CN(close, timeperiod):
    diff = map(lambda x, y: x - y, close[1:], close[:-1])
    diffGt0 = map(lambda x: 0 if x < 0 else x, diff)
    diffABS = map(lambda x: abs(x), diff)
    diff = np.array(diff)
    diffGt0 = np.array(diffGt0)
    diffABS = np.array(diffABS)
    diff = np.append(diff[0], diff)
    diffGt0 = np.append(diffGt0[0], diffGt0)
    diffABS = np.append(diffABS[0], diffABS)
    rsi = map(lambda x: SMA_CN(diffGt0[:x], timeperiod) / SMA_CN(diffABS[:x], timeperiod) * 100
              , range(1, len(diffGt0) + 1))

    return np.array(rsi)


# 同花顺和通达信等软件中的MACD
def MACD_CN(close, fastperiod, slowperiod, signalperiod):
    macdDIFF, macdDEA, macd = tl.MACDEXT(close, fastperiod=fastperiod, fastmatype=1, slowperiod=slowperiod,
                                         slowmatype=1, signalperiod=signalperiod, signalmatype=1)
    macd = macd * 2
    return macdDIFF, macdDEA, macd


def kdj_filter(code):
    for s in code.index:
        hData = attribute_history(s, 30, unit='1d'
                                  , fields=('close', 'volume', 'open', 'high', 'low')
                                  , skip_paused=True
                                  , df=False)
        volume = hData['volume']
        volume = np.array(volume, dtype='f8')
        close = hData['close']
        open = hData['open']
        high = hData['high']
        low = hData['low']

        kValue, dValue, jValue = KDJ_CN(high, low, close, 9, 3, 3)

        print(kValue[-1], dValue[-1], jValue[-1], jValue[-2])
        if jValue[-1] < jValue[-2]:
            print(s, 'drop---------------------------------------------------')
            code.drop([s], inplace=True)

    return code