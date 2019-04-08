# coding: utf-8
from jqdatasdk import *
import jqdata

def get_stocks_pool(code):
    code = get_index_stocks(code)
    code = stocks_filter(code)
    return code

def stocks_filter(code):
    current_data = get_current_data(code)                    #stocks data
    code = [s for s in code if not current_data[s].is_st]  # 过滤ST
    code = [s for s in code if not current_data[s].paused]  # 过滤停牌
    code = [s for s in code if not current_data[s].day_open >= current_data[s].high_limit]  # 过滤涨停
    code = [s for s in code if not current_data[s].day_open <= current_data[s].low_limit]  # 过滤跌停
    return code