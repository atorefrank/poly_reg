import pandas as pd
import talib


def Pivot_high(values,lb,rb):
    mb =lb+rb+1
    # print(mb)
    df_temp = pd.DataFrame({'values':values})
    
    df_temp['max_ind'] =  talib.MAXINDEX(df_temp['values'],mb)-df_temp['values'].expanding(1).count()+1
    df_temp['max_2'] =  talib.MAXINDEX(df_temp['values'],lb)-df_temp['values'].expanding(1).count()+1
    df_temp['final'] = 0

    print(df_temp)
    
    # print(values)
    max_bool = df_temp['max_ind']== -lb
    # print(df_temp.loc[max_bool,'final'])
    # print(df_temp['values'].shift(lb))
    print('===============')
    # df_temp.loc[df_temp['max_ind']==-lb,'final'] = df_temp['values'].shift(lb)
    df_temp.loc[df_temp['max_ind']== -lb,'final']
    print(df_temp)
    
    # df_temp_2 = df_temp['values'].shift(lb)
    return df_temp['final']

def pivot_high(values,lb,rb):
    mb =lb+rb+1
    df_temp = pd.DataFrame({'values':values})
    df_temp['max_ind'] =  talib.MAXINDEX(df_temp['values'],mb)-df_temp['values'].expanding(1).count()+1
    df_temp['max_2'] =  talib.MAXINDEX(df_temp['values'],lb)-df_temp['values'].expanding(1).count()+1
    df_temp['final'] = 0
    
    df_temp.loc[df_temp['max_ind']==-lb,'final'] = df_temp['values'].shift(lb)
    # print(df_temp)
    return df_temp['final']





def pivot_low(values,lb,rb):
    mb =lb+rb+1
    df_temp = pd.DataFrame({'values':values})
    df_temp['max_ind'] =  talib.MININDEX(df_temp['values'],mb)-df_temp['values'].expanding(1).count()+1
    df_temp['max_2'] =  talib.MININDEX(df_temp['values'],lb)-df_temp['values'].expanding(1).count()+1
    df_temp['final'] = 0
    df_temp.loc[df_temp['max_ind']==-lb,'final'] = df_temp['values'].shift(lb)
    return df_temp['final']