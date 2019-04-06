import talib

def getMACD( data ):
   close = data["close"]
   DIFF, DEA, MACD = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
   data['DIFF'] = DIFF
   data['DEA'] = DEA
   data['MACD'] = MACD * 2
   return data

def getKDJ( data ):
   high = data["high"]
   low = data["low"]
   close = data["close"]
   K, D = talib.STOCH(high, low, close, fastk_period=9, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
   data['KDJ_K'] = K
   data['KDJ_D'] = D
   data['KDJ_J'] = 3*D - 2*K
   return data

def getRSI( data ):
   close = data["close"]
   RSI6 = talib.RSI(close, timeperiod=6)
   RSI12 = talib.RSI(close, timeperiod=12)
   RSI24 = talib.RSI(close, timeperiod=24)

   data['RSI6'] = RSI6
   data['RSI12'] = RSI12
   data['RSI24'] = RSI24
   return data

def getDMI( data ):
   high = data["high"]
   low = data["low"]
   close = data["close"]

   DI1 = talib.PLUS_DI(high, low, close, timeperiod=14)
   DI2 = talib.MINUS_DI(high, low, close, timeperiod=14)
   ADX = talib.ADX(high, low, close, timeperiod=14)
   ADXR =talib.ADXR(high, low, close, timeperiod=14)

   data['DMI_DI+'] = DI1
   data['DMI_DI-'] = DI2
   data['ADX'] = ADX
   data['ADXR'] = ADXR
   return data

