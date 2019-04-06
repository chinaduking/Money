def MacdQuality(data):
    try:
        df =data[['trade_date','MACD']].sort_index(ascending=False).head(30)
    except:
        print('传入的参数中不包含MACD,ERR')

    print(df)