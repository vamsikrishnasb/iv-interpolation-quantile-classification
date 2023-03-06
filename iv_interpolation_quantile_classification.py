from nsepython import *
import pandas as pd
from scipy.stats import norm
from scipy import optimize
from math import sqrt, log
import numpy as np
import time
import os
import glob
import warnings
from jugaad_data.nse import index_csv, index_df
from datetime import date
import plotly
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

warnings.simplefilter("ignore")

# last_date = '2010-11-01'
date1 = '2020-01-01'
date2 = '2023-02-28'
xarray_date1 = '2010-11-01'
xarray_date2 = date2

folder_path = 'your/folder/path/here'
path = folder_path + '/sample_input_1.csv'
df = pd.read_csv(path)
df = df[(df['days_to_expiry'].between(1, 366))].reset_index().drop('index', 1)

dates = df[df['date'].between(date1, date2)]['date'].unique()
for date in dates:
    i_var_df = pd.DataFrame(columns=['date', 'moneyness_close'])
    expiries = df[(df['date'] == date)]['expiry'].unique()
    dte = df[(df['date'] == date)]['days_to_expiry'].unique()
    dtes = sorted(dte)
    moneyness = []
    moneyness_list = list(reversed(np.arange(-0.4000, 0.4001, 0.0001)))
    temp = pd.DataFrame()
    temp['temp'] = moneyness_list
    temp['temp'] = temp.sort_values(by='temp', ascending=False)
    temp['temp'] = round(temp['temp'], 4)
    moneyness_list = temp['temp'].tolist()
    i_var_df['moneyness_close'] = moneyness_list
    i_var_df['date'] = date
    i_var_df = i_var_df[['date', 'moneyness_close']]
    for expiry, dte in zip(expiries, dtes):
        options_df = df[
            (df['date'] == date) 
            & (df['expiry'] == expiry)
        ][[
            'date', 'strike', 'expiry', 'otm_option_type', 'moneyness_close', 
            'days_to_expiry', 'put_no_of_contracts', 'put_turnover_in_lacs_option', 
            'put_premium_turnover_in_lacs', 'put_open_interest', 'put_change_in_oi', 
            'call_no_of_contracts', 'call_turnover_in_lacs_option', 
            'call_premium_turnover_in_lacs', 'call_open_interest', 'call_change_in_oi', 
            'implied_volatility', 'forward_close'
        ]]
        avg_put_oi = options_df['put_open_interest'].quantile(0.25)
        avg_call_oi = options_df['call_open_interest'].quantile(0.25)
        options_df = options_df[(
            ((options_df['otm_option_type'] == 'PE') 
             & (options_df['put_open_interest'] >= avg_put_oi))
            | ((options_df['otm_option_type'] == 'CE') 
             & (options_df['call_open_interest'] >= avg_call_oi))
        )][['moneyness_close', 'implied_volatility']]
        options_df['moneyness_close'] = round(options_df['moneyness_close'], 4)
        options_df = options_df.reset_index().drop('index', 1)
        i_var_df = i_var_df.merge(options_df, on=['moneyness_close'], how='left')
        i_var_df = i_var_df.rename(columns={
            'implied_volatility': dte
        })
        i_var_df[dte] = i_var_df[dte]**2
        implied_variance = pd.Series(i_var_df[dte]).interpolate(limit_area='inside')
        i_var_df[dte] = implied_variance
        
    dte_list = list(range(1, 367))
    for dte in dte_list:
        if dte not in i_var_df.columns:
            i_var_df[dte] = np.nan
    date_temp = i_var_df.pop('date')
    moneyness_close_temp = i_var_df.pop('moneyness_close')
    i_var_df = i_var_df.reindex(sorted(i_var_df.columns), axis=1)
    i_var_df.insert(0, 'date', date_temp)
    i_var_df.insert(1, 'moneyness_close', moneyness_close_temp)
    
    # Vertical interpolation
    dte_list = list(range(1, 367))
    for dte in dte_list:
        implied_variance = pd.Series(i_var_df[dte]).interpolate(limit_area='inside')
        i_var_df[dte] = implied_variance
    
    # Horizontal interpolation through transpose
    date_temp = i_var_df.pop('date')
    moneyness_close_temp = i_var_df.pop('moneyness_close')
    df_len = len(i_var_df)
    i_var_df = i_var_df.transpose()
    for i in range(0, df_len):
        implied_variance = pd.Series(i_var_df[i]).interpolate(limit_area='inside')
        i_var_df[i] = implied_variance ** 0.5
    i_var_df = i_var_df.transpose()
    i_var_df.insert(0, 'date', date_temp)
    i_var_df.insert(1, 'moneyness_close', moneyness_close_temp)
#     i_var_df.to_csv('/Users/vamsikrishnasb/My Drive/Financial Analysis/backtesting/IVol PC Values/Date Wise IVol Surface/Detailed/'+date+'.csv', index=False)
    reduced_moneyness_list = list(reversed(np.arange(-0.1000, 0.1001, 0.0005)))
    temp = pd.DataFrame()
    temp['temp'] = reduced_moneyness_list
    temp['temp'] = temp.sort_values(by='temp', ascending=False)
    temp['temp'] = round(temp['temp'], 4)
    reduced_moneyness_list = temp['temp'].tolist()
    i_var_reduced_df = i_var_df[i_var_df['moneyness_close'].isin(reduced_moneyness_list)].reset_index().drop('index', 1)
    i_var_reduced_df.to_csv('/to/your/desired/folder/data.csv', index=False)
    
import xarray as xr
xarray_dates = df[df['date'].between(xarray_date1, xarray_date2)]['date'].unique()
path = '/to/your/desired/folder/'+xarray_dates[0]+'.csv'
df = pd.read_csv(path)
ds = df.to_xarray()
for i in range(1, len(xarray_dates)):
    path = '/to/your/desired/folder/'+xarray_dates[i]+'.csv'
    df2 = pd.read_csv(path)
    ds2 = df2.to_xarray()
    ds = xr.concat([ds,ds2], dim='new_index')
df_final = ds.to_dataframe()
df_final.to_csv('/o/your/desired/folder/'+xarray_date2+'.csv')

path = '/to/your/desired/folder/'+xarray_date2+'.csv'
df_final = pd.read_csv(path, index_col=[0,1], skipinitialspace=True)
percentile_list = list(np.arange(5, 101, 5))
pc_df = pd.DataFrame()
pc_df = df_final.groupby('moneyness_close').quantile(q = 0, interpolation="nearest").reset_index()
pc_df['pc'] = 0
pc_temp = pc_df.pop('pc')    
pc_df.insert(0, 'pc', pc_temp)
for pc in percentile_list:
    temp_df = df_final.groupby('moneyness_close').quantile(q = 1.00 * pc / 100, interpolation="nearest").reset_index()
    temp_df['pc'] = pc
    pc_temp = temp_df.pop('pc')    
    temp_df.insert(0, 'pc', pc_temp)
    pc_df = pd.concat([pc_df,temp_df], axis=0).reset_index().drop('index', 1)
pc_df.to_csv('/to/your/desired/folder/'+xarray_date2+'.csv', index=False)