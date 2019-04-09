# 克隆自聚宽文章：https://www.joinquant.com/post/19402
# 标题：林奇PEG选股轮动，十年20倍，有止损模块
# 作者：一梦春秋

'''
策略思路：
选股：林奇PEG，PE/G排序，G=EPS增长率
择时：无
持仓：轮动

'''

##### 下方代码为 IDE 运行必备代码 #####
if __name__ == '__main__':
    import jqsdk
    params = {
        'token':'55d808531808431e12680bc3d317be6d',
        'algorithmId':1,
        'baseCapital':1000000,
        'frequency':'day',
        'startTime':'2019-01-01',
        'endTime':'2019-04-08',
        'name':"Test1",
    }
    jqsdk.run(params)


# 导入函数库
import statsmodels.api as sm
# from pandas.stats.api import ols

# 初始化函数，设定基准等等
def initialize(context):
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 过滤掉order系列API产生的比error级别低的log
    # log.set_level('order', 'error')
    ### 股票相关设定 ###
    set_parameter(context)
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')
    
    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
      # 开盘前运行
    run_daily(before_market_open, time='before_open', reference_security='000300.XSHG') 
      # 开盘时运行
    # run_daily(market_open, time='open', reference_security='000300.XSHG')
      # 收盘后运行
    run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')
    
    trade_func(context)
    #每月轮动
    # run_monthly(trade_func, 1, "09:30")
    #每周轮动
    run_weekly(trade_func, 1, "09:30")
'''
==============================参数设置部分================================
'''
def set_parameter(context):
    #持仓股票数
    g.stock_num = 10
    g.stock_pos = 0
    g.danger = False
    g.danger_days = 0
    g.maxCash = 0
    
    #风险参考基准
    g.security = '000300.XSHG'
    # 设定策略运行基准
    set_benchmark(g.security)
    #记录策略运行天数
    g.days = 0
    
    
## 开盘前运行函数     
def before_market_open(context):
    # 输出运行时间
    #log.info('函数运行时间(before_market_open)：'+str(context.current_dt.time()))
    g.days += 1
    # 给微信发送消息（添加模拟交易，并绑定微信生效）
    log.info('策略正常，运行第%s天~'%g.days)
    send_message('策略正常，运行第%s天~'%g.days)

## 开盘时运行函数
def handle_data(context, data):
    # record(cash=context.portfolio.cash)
    
    ma1 = data['000001.XSHG'].mavg(1, 'close')
    ma5 = data['000001.XSHG'].mavg(5, 'close')
    ma10 = data['000001.XSHG'].mavg(10, 'close')
    ma20 = data['000001.XSHG'].mavg(20, 'close')
    ma30 = data['000001.XSHG'].mavg(30, 'close')
    ma60 = data['000001.XSHG'].mavg(60, 'close')
    
    if ma1 > ma30:
        g.danger = False
        # order_target_value('511880.XSHG',0)
    elif ma1 < ma30:
        g.danger = True
        #清仓
        for security,v in context.portfolio.positions.items():
            order_target(security, 0)


            
            
#策略选股买卖部分    
def trade_func(context):
    #大盘风险
    if g.danger:
        return
    
    #获取股票池
    df = get_fundamentals(
        query(
            valuation.code,
            valuation.pb_ratio,
            indicator.roe,
            valuation.pe_ratio,
            indicator.eps
        )
        .filter(
            indicator.eps > 0,
            valuation.pb_ratio > 0,
            valuation.pe_ratio > 0
        )
    )
    df.index = df['code'].values
    
    #上一年的EPS
    current_dt = context.current_dt
    last_year_date = current_dt + timedelta(days=-365)
    last_year_df = get_fundamentals(
        query(
            valuation.code,
            valuation.pb_ratio,
            indicator.roe,
            valuation.pe_ratio,
            indicator.eps
        )
        .filter(
            indicator.eps > 0,
            valuation.pb_ratio > 0,
            valuation.pe_ratio > 0
        )
        ,date=last_year_date
    )
    last_year_df.index = last_year_df['code'].values
    df['last_year_eps']=last_year_df['eps']
    df['eps_grouth']=df['eps'] - df['last_year_eps']
   
    
    #进行盈利>0筛选
    df = df[df['eps_grouth']>0]
    df = df[(df['eps']>0) & (df['pe_ratio']>0)]
    
    #G=eps增长率
    df['g']=df['eps_grouth'] / df['last_year_eps']
    df['peg']=df['pe_ratio'] / df['g'] / 100
    
    #取排名top N的股票 进行轮动
    # df = df.sort('peg', ascending=True)[:g.stock_num]
    df = df.sort_values(by = 'peg',axis = 0,ascending=True)[:g.stock_num]    #python3
    # df = df.sort(columns = 'peg',axis = 0,ascending=True)[:g.stock_num]    #python2

    #过滤停牌 ST
    pool = df.index
    pool = paused_filter(pool)
    pool = delisted_filter(pool)
    pool = st_filter(pool)
    
    log.info('总共选出%s只股票'%len(pool))
    if len(pool) == 0:
        return;

    #得到每只股票应该分配的资金
    cash = context.portfolio.total_value/len(pool)
    #获取已经持仓列表
    hold_stock = context.portfolio.positions.keys() 
    #卖出不在持仓中的股票
    for s in hold_stock:
        if s not in pool:
            order_target(s,0)
    #买入股票
    for s in pool:
        log.info("买入股票:"+s)
        order_target_value(s,cash)
        
#打分工具
def f_sum(x):
    point = sum(x)
    return point
        
## 收盘后运行函数  
def after_market_close(context):
    #得到当天所有成交记录
    trades = get_trades()
    for _trade in trades.values():
        log.info('成交记录：'+str(_trade))
    #打印账户总资产
    log.info('今日账户总资产：%s'%round(context.portfolio.total_value,2))
    log.info('##############################################################')

# 过滤停牌、退市、ST股票
def paused_filter(security_list):
    current_data = get_current_data()
    security_list = [stock for stock in security_list if not current_data[stock].paused]
    return security_list


def delisted_filter(security_list):
    current_data = get_current_data()
    security_list = [stock for stock in security_list if not '退' in current_data[stock].name]
    return security_list

def st_filter(security_list):
    current_data = get_current_data()
    security_list = [stock for stock in security_list if not current_data[stock].is_st]
    return security_list

#------------------------------Tactics--因子----------------------------#
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

# CMV:流通市值,越小越好
def Rank_by_CMV(code):
    #获取因子
    q = query(
        valuation.code,
        valuation.circulating_market_cap,
    ).filter(
        valuation.code.in_(code)
    )
    df = get_fundamentals(q)
    df.index = df['code']
    df.columns = ['code'] + ['CMV']

    # 升序排序
    df = df.sort_values(by = 'CMV',axis = 0,ascending=True)    #python3
    # df = df.sort(columns = 'CMV',axis = 0,ascending=True)    #python2
    return df

#------------------User Api---------------------------#
def get_stocks_pool(code):
    code = get_index_stocks(code)
    # code = stocks_filter(code)
    return code

def stocks_filter(code):
    current_data = get_current_data(code)                  #stocks data
    print(current_data)
    code = [s for s in code if not current_data[s].is_st]  # 过滤ST
    code = [s for s in code if not current_data[s].paused]  # 过滤停牌
    code = [s for s in code if not current_data[s].day_open >= current_data[s].high_limit]  # 过滤涨停
    code = [s for s in code if not current_data[s].day_open <= current_data[s].low_limit]  # 过滤跌停
    return code