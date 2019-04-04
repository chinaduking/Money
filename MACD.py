import pandas as pd
import tushare as ts
#定义一个函数get_a_share
def get_a_share(code):
    #使用tushare的get_k_data调用A股股票数据
    data = ts.get_k_data(code)
    #删除列code
    # data = data.drop('code',axis = 1)
    #返回修改后的股票数据
    return data


#定义函数，获取macd,导入数据，初始化三个参数
def get_macd_data(data,short=0,long=0,mid=0):
    if short==0:
        short=12
    if long==0:
        long=26
    if mid==0:
        mid=9
    #计算短期的ema，使用pandas的ewm得到指数加权的方法，mean方法指定数据用于平均
    data['sema']=pd.Series(data['close']).ewm(span=short).mean()
    #计算长期的ema，方式同上
    data['lema']=pd.Series(data['close']).ewm(span=long).mean()
    #填充为na的数据
    data.fillna(0,inplace=True)
    #计算dif，加入新列data_dif
    data['data_dif']=data['sema']-data['lema']
    #计算dea
    data['data_dea']=pd.Series(data['data_dif']).ewm(span=mid).mean()
    #计算macd
    data['data_macd']=2*(data['data_dif']-data['data_dea'])
    #填充为na的数据
    data.fillna(0,inplace=True)
    #返回data的三个新列
    # return data
    return data[['date','data_dif','data_dea','data_macd']]

#请输入A股股票代码
code = str(input('输入A股股票代码:'))
data = get_a_share(code)
macd = get_macd_data(data)
print(macd)