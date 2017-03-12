# -*- coding: utf-8 -*-
"""
Portfolio Optimization in Python using yahoo finance data
/Bi-annual rebalancing on the way?/

Created on Fri Oct 28 10:50:00 2016
@author: Judah Okwuobi, 
"""

import os
proj_dir = "C:\\Users\\d_smo\\OneDrive\\Life\\Learning\\1. UBC\\Courses 2013-2018\\Mathematics\\"
proj_dir = proj_dir + "Math 441 - Mathematical Modeling - Discrete Optimization Problems\\Term Project\\"
proj_pyfl = proj_dir + "math441project\\Python Files" 

# Changing directory to project folder
os.chdir(proj_pyfl) 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib
import datetime as dt
from scipy import stats 
#import yahoo_finance as yahoo
#from pprint import pprint

###########################################################
## Importing the Quotes From Yahoo
###########################################################

base_url = "http://ichart.finance.yahoo.com/table.csv?s="
def make_url(ticker_symbol):
    # ticker_symbol is a string
    return base_url + ticker_symbol

output_path = proj_dir + "Stock Data"
def make_filename(ticker_symbol, directory):
    # We should probably handle the case of the directory not being there
    return output_path + "\\" + directory + "\\" + ticker_symbol + ".csv"

def pull_historical_data(symbol_list, directory):
    # symbol_list is a list of strings
    for ticker_symbol in symbol_list:
        try:
            urllib.request.urlretrieve(make_url(ticker_symbol), 
                                       make_filename(ticker_symbol, directory))
#        except urllib.ContentTooShortError as e:
#            outfile = open(make_filename(ticker_symbol, directory), "w")
#            outfile.write(e.content)
#            outfile.close()
        except ValueError:
            print("Don't really know what went wrong")        
            
# Obtain Quotes in desired directory
stock_list = ["AAPL", "GOOG", "NFLX", "MSFT", "GE", "F"]
pull_historical_data(stock_list, "Yahoo Finance")  

# Store Data in dataframes 
data = {} 
for stock in stock_list:
    data_dir = make_filename(stock, "Yahoo Finance")
    data[stock] = pd.read_csv(data_dir, header = 0, 
                                    index_col = 0,
                                    # Infer date time significantly speeds up import time
                                    infer_datetime_format = True)
                             
                                        

start_date = "2000-01-01"
end_date = "2016-01-01"
data_train = {}   
for stock in stock_list:
    # Subsetting the matrix to desired preriod
    data_train[stock] = data[stock].ix[end_date:start_date]
    # Sorting for better sequence handling
    data_train[stock] = data_train[stock].sort_index()
    del stock

##########################################################
# Building the Optimization Model
##########################################################


   
# Calculating the Returns & Inserting it into original data frame
for stock in stock_list:
    # Inserting  daily returns as (1+r) rather than r
    data_train[stock].loc[:,("Adj Return")] = 1 + data_train[stock].loc[:,("Adj Close")].pct_change()
    data_train[stock].loc[0:1,("Adj Return")]= 1
    #data_train[stock].loc[:,("Close")]/data_train[stock].loc[:,("Open")] 
    # Come back to fix this to be actuall run on adj close column    
    del stock 

## TODO:
## Anualize the returns and ouput average over years in consideration
def annual_ret(df, yr_rng):
    # df is a data frame
    # yr_rng is a list of two strings which are start and end dates
    
    #Monthly Returns
    df_month = df.resample('BM', how=lambda x: x[-1])
    df_month.loc[:, ("Adj Return")] = df_month.loc[:,("Adj Close")].pct_change()
    df_month.loc[0:1,("Adj Return")]= 1    
    
    
    start = yr_rng[0]
    end   = yr_rng[1]
    lower = start
    #Convoluted form but works on all years I believe
    upper = str(int(lower[0:4]) + 1) + lower[4:] 
    geo_means = []    
    
        
    while int(lower[0:4]) < int(end[0:4]):
        #print(lower)
        #print(upper)
        geo_means.append(stats.gmean(df.loc[lower:upper,("Adj Return")]))
        lower =str(int(lower[0:4]) + 1) + lower[4:]
        upper = str(int(lower[0:4]) + 1) + lower[4:] 
    #Calculate the average over the years for an aver
    print(geo_means)
    avg = sum(geo_means)/len(geo_means)
    return avg

x = annual_ret(data_train["AAPL"], ["2000-01-01","2012-01-01"])
print(x)

data_train["GE"].loc[:,("Adj Return")]
## Average the Anualized returns
## Input them into a dictionary for each of the stocks using stocklist
"""
End result
{AAPL: [rt_1, ret 2]}
"""

   