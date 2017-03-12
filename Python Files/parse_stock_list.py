import csv
import sys
import pickle
import argparse
import datetime
import pandas as pd
import pandas_datareader.data as web
import pandas_datareader._utils as web_util

STOREFILENAME = 'stockdb.h5'
store = pd.HDFStore(STOREFILENAME, 'a')
beginning = datetime.datetime(1890, 1, 1)
list_of_symbols = []
def parse_list(file_name):
    los = []
    with open(file_name) as company_list_file:
        company_file_reader = csv.DictReader(company_list_file)
        for row in company_file_reader:
            symbol = row['Symbol']
            los.append(symbol)
    return los


def download_symbols(file_name, los, suffix=''):
    failed_tickers = []
    final_symbol_list = []
    with open(file_name, 'wb') as symbol_file:
        for stock in los:
            try:
                stock_data = web.DataReader(stock, 'yahoo', beginning)
                store.put(stock, stock_data)
                print(stock)
                final_symbol_list.append(stock)
            except web_util.RemoteDataError:
                if suffix: 
                    stock = stock + suffix
                    try:
                        stock_data = web.DataReader(stock, 'yahoo', beginning)
                        store.put(stock, stock_data)
                        print(stock)
                        final_symbol_list.append(stock)
                    except web_util.RemoteDataError:
                        pass
                pass
        final_symbol_list.sort()
        symbol_file.truncate()
        pickle.dump(final_symbol_list, symbol_file)
        symbol_file.close()
    return final_symbol_list

sp_file_to_parse = 'sp_company_list.csv'
sp_symbol_file = 'sp_symbol_list'

tsx_file_to_parse = 'TSX_300_List.csv'
tsx_symbol_file = 'tsx_symbol_list'

list_of_symbols = parse_list(sp_file_to_parse)
print("DOWNLOADING SP LIST")
sp_final_symbol_list = download_symbols(sp_symbol_file, list_of_symbols)

del list_of_symbols

list_of_symbols = parse_list(tsx_file_to_parse)
print("DOWNLOADING TSX LIST")
tsx_final_symbol_list = download_symbols(tsx_symbol_file, list_of_symbols, '.TO')

final_symbol_list = sp_final_symbol_list + tsx_final_symbol_list

with open('sp_tsx_symbol_list', 'wb') as symbol_file:
    symbol_file.truncate()
    pickle.dump(final_symbol_list, symbol_file)


store.close()

