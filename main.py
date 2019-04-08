# coding: utf-8

##### 下方代码为 IDE 运行必备代码 #####
if __name__ == '__main__':
    import jqsdk
    params = {
        'token':'d1e63b419abd774910d1d76cf0f3a337',
        'algorithmId':1,
        'baseCapital':1000000,
        'frequency':'day',
        'startTime':'2019-01-01',
        'endTime':'2019-04-08',
        'name':"Test1",
    }
    jqsdk.run(params)

##########################################################################
import jqdata
import scipy.optimize as sco
from kuanke.wizard import *
import numpy as np
import talib as tl

###########################################################################
def initialize(context):
    # 参数设置
    set_params(context)
    # 设置回测的参数
    set_backtest()

    # 每天运行
    run_daily(DayTactics, '8:00')
    # # 每周运行
    # run_weekly(WeeklyTactics, -1, '20:00')
    # # 每月运行
    # run_monthly(MonthlyTactics, -1, time='20:00')

#####################################################################
#回测时函数
def handle_data(context, data):

    # day = context.current_dt.day
    # if day > 24:
    #     for stock in g.buy_stock:
    #         order_target_value(stock, 0)
    #     return

    # sell_stocks(context)

    # 指数止损
    # if conduct_dapan_stoploss(context, '000001.XSHG', 4, -0.05):
    #     return

    # buy_stocks(context)

    log.info('(MonthlyTactics--market_open):' + str(context.current_dt.time()))

######################################################################
def set_params(context):
    # 需要购买股票的股票池
    g.buy_stock = []
    g.mcad_isring = False

#回测设置
def set_backtest():
    # 使用真是数据进行回测
    set_option('use_real_price', True)
    #设置最大持仓股票数量
    g.MaxCount = 5

####################################################################
def DayTactics(context):
    g.mcad_isring = conduct_dapan_MACD()
    if g.mcad_isring == True:
        if context.portfolio.positions.keys() != []:
            return
        WeeklyTactics(context)
        buy_stocks(context)
        print('buy...............')
    else:
        if context.portfolio.positions.keys() == []:
            return

        for stock in g.buy_stock:
            order_target_value(stock, 0)
        print('sell.................')

def WeeklyTactics(context):
    code = get_stocks_pool('000001.XSHG')
    df = Rank_by_FAP(code)
    g.buy_stock = list(df.head(5).index)

def MonthlyTactics(context):
    log.info('(MonthlyTactics--market_open):' + str(context.current_dt.time()))


#------------------User Api---------------------------#
def get_stocks_pool(code):
    code = get_index_stocks(code)
    code = stocks_filter(code)
    return code

def stocks_filter(code):
    current_data = get_current_data(code)                  #stocks data
    code = [s for s in code if not current_data[s].is_st]  # 过滤ST
    code = [s for s in code if not current_data[s].paused]  # 过滤停牌
    code = [s for s in code if not current_data[s].day_open >= current_data[s].high_limit]  # 过滤涨停
    code = [s for s in code if not current_data[s].day_open <= current_data[s].low_limit]  # 过滤跌停
    return code

# 清仓不在股票池里的股票
def sell_stocks(context):
    if context.portfolio.positions.keys() != []:
        for stock in context.portfolio.positions.keys():
            if stock not in g.buy_stock:
                order_target_value(stock, 0)

# 买入股票池里的股票
def buy_stocks(context):
    num = len(g.buy_stock)
    if num == 0:
        return 0

    df = history(400, '1d', 'close', g.buy_stock, df=True)
    g.returns = np.log(df / df.shift(1))

    # 约束是所有参数(权重)的总和为1。这可以用minimize函数的约定表达如下
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # 我们还将参数值(权重)限制在0和1之间。这些值以多个元组组成的一个元组形式提供给最小化函数
    bnds = tuple((0, 1) for x in range(num))

    # 优化函数调用中忽略的唯一输入是起始参数列表(对权重的初始猜测)。我们简单的使用平均分布。
    optv = sco.minimize(min_variance, num * [1. / num, ], method='SLSQP', bounds=bnds, constraints=cons)
    print(optv['x'].round(3))

    port_value = context.portfolio.portfolio_value
    for stock in g.buy_stock:
        buyValue = (optv['x'][g.buy_stock.index(stock)] * port_value)
        order_target_value(stock, buyValue)

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

# 同花顺和通达信等软件中的MACD
def MACD_CN(close, fastperiod, slowperiod, signalperiod):
    macdDIFF, macdDEA, macd = tl.MACDEXT(close, fastperiod=fastperiod, fastmatype=1, slowperiod=slowperiod,
                                         slowmatype=1, signalperiod=signalperiod, signalmatype=1)
    macd = macd * 2
    return macdDIFF, macdDEA, macd

def conduct_dapan_MACD():
    dapan = attribute_history('000001.XSHG', 100, '1d','close',df=False,fq='pre')
    print(dapan['close'])
    macdDIFF, macdDEA, macd = MACD_CN(dapan['close'], 12, 26, 9)
    record(macdDIFF=macdDIFF[-1], macdDEA=macdDEA[-1], macd=macd[-1])

    if macdDEA[-1] - macdDEA[-2] > 0:
        return True
    else:
        return False

#------------------------------Tactics------------------------------#
# FAP:固定资产/总资产 比率,越小越好
def Rank_by_FAP(code):
    #获取 固定资产/总资产 因子
    q = query(
        valuation.code,
        balance.fixed_assets / balance.total_assets,
    ).filter(
        valuation.code.in_(code)
    )
    df = get_fundamentals(q)
    df.index = df['code']
    df.columns = ['code'] + ['FAP']

    # 升序排序
    df = df.sort_values(by = 'FAP',axis = 0,ascending=True)    #python3
    # df = df.sort(columns = 'FAP',axis = 0,ascending=True)    #python2
    return df