import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

s = pd.Series([1, 2, np.nan, 4, 5])
print(s)
print(s.name)
print(s.index)

print(s.iloc[0])
print(s.iloc[0:5:1])

new_index = pd.date_range("20160101",periods=len(s), freq="D")
s.index = new_index;
print(s)

print(s.loc[(s < 3) & (s > 1)])

data = pd.read_excel('sz50.xlsx', sheetname=0, index_col='datetime')
print(data)

Series = data.close
print(Series.head())

monthly_prices = Series.resample('M').last()
print(monthly_prices.head(5))

from datetime import datetime
data_s= Series.loc[datetime(2017,1,1):datetime(2017,1,10)]
data_r=data_s.resample('D').mean() #插入每一天
print(data_r.head(10))

print(data_r.head(10).dropna())  #去掉缺失值

print(data_r.head(10).fillna(method='ffill'))  #填写缺失的天为前一天的价格。

print(data_r.head(10).fillna(method='bfill'))  #填写缺失的天为前一天的价格。