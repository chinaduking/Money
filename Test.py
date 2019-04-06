import StockIndex as si
import Stock as s
import TacticsQuality as tq
import matplotlib.pyplot as plt

#请输入A股股票代码
# code = str(input('输入A股股票代码:'))
code = 600030
data = s.getStockDay(code)

# print(data)
df = si.getMACD(data)
print(df)
tq.MacdQuality(df)

# print(si.getKDJ(data))
# print(si.getRSI(data))
# print(si.getDMI(data))

# plt.title('DMI')
# plt.plot( data['date'],data['DMI_DI+'], color='yellow', label='DI+')
# plt.plot(data['date'],data['DMI_DI-'], color='red', label='DI-')
# plt.plot(data['date'],data['ADX'], color='green', label='ADX')
# plt.plot(data['date'],data['ADXR'], color='blue', label='ADXR')
# plt.legend() # 显示图例
# plt.show()
