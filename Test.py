import tushare as ts
# 80ef13f73715a599af51122b7edfdd8dbac50543f5c909aaf4ebe567
print(ts.__version__)

ts.set_token('80ef13f73715a599af51122b7edfdd8dbac50543f5c909aaf4ebe567')

pro = ts.pro_api()

df = pro.trade_cal(exchange='', start_date='20180901', end_date='20181001', fields='exchange,cal_date,is_open,pretrade_date', is_open='0')

print(df)