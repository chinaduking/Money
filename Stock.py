import struct
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
def getStockDay(code):
    global preClose
    try:
        try:
            ofile = open("D:/tongdaxin/vipdoc/sh/lday/sh" + str(code) + ".day", 'rb')
        except:
            ofile = open("D:/tongdaxin/vipdoc/sz/lday/sz" + str(code) + ".day", 'rb')
    except:
        print('请输入正确的股票代码')
        exit()

    buf = ofile.read()
    ofile.close()
    num = len(buf)
    no = num / 32
    b = 0
    e = 32
    items = list()
    for i in range(int(no)):
        a = struct.unpack('IIIIIfII', buf[b:e])
        year = str(int(a[0] / 10000))
        m = int((a[0] % 10000) / 100)
        month = str(m).zfill(2)  # 补零函数
        d = (a[0] % 10000) % 100
        day = str(d).zfill(2)
        dd = str(year) + month + day
        openprice = a[1] / 100.0
        high = a[2] / 100.0
        low = a[3] / 100.0
        close = a[4] / 100.0
        amount = a[5] / 10.0
        vol = a[6]
        unused = a[7]
        if i == 0:
            preClose = close
        ratio = round((close - preClose) / preClose * 100, 2)
        preClose = close
        item = [code, dd, openprice, high, low, close, ratio, amount, vol]
        items.append(item)
        b = b + 32
        e = e + 32
    frame_data = pd.DataFrame(items)
    frame_data.columns = ['ts_code', 'trade_date','open', 'high','low', 'close','ratio', 'amount','vol']
    return frame_data