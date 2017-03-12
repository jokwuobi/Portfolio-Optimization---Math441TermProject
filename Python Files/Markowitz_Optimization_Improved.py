"""
Portfolio Optimization in Python using yahoo finance data
/Bi-annual rebalancing on the way?/

Created on Fri Oct 28 10:50:00 2016
@author: Judah Okwuobi, Rohin Patel, Brandon Loss
"""

###########################################################
## Importing the Quotes From Yahoo
###########################################################

import os
import sys
import argparse
import pickle
import pdb
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import urllib
#import datetime as dt
import cvxpy as cvx
from scipy import stats 
#from pandas_datareader import data as web
#import pandas.io.data as web #OLD VERSION
import pandas_datareader.data as web
import pandas_datareader._utils as web_util
import datetime
import matplotlib.pyplot as plt
#matplotlib inline
#config InlineBackend.figure_format = 'svg'

proj_dir = "C:\\Users\\d_smo\\OneDrive\\Life\\Learning\\1. UBC\\Courses 2013-2018\\Mathematics\\"
proj_dir = proj_dir + "Math 441 - Mathematical Modeling - Discrete Optimization Problems\\Term Project\\"
proj_pyfl = proj_dir + "math441project\\Python Files"
proj_savefig = proj_dir + "math441project\\Figures\\" + "StockMovement.jpg"

STOREFILENAME = 'stockdb.h5'

##########################################################
# Building the Optimization Model
##########################################################

## Function to anualize the returns and ouput geometric average over years in consideration

print("Preparing data.....\n")
def annual_ret(df, start, end, s):
    # df is a data frame
    # start & end are date ranges for stock returns
    
    #Monthly Returns
    ## first argument is frequency of the rescaling BQ = business quarterly
    try:
        df_q = df.resample('A')#.mean()#apply(lambda x: x[-1])
        df_q.loc[:, ("Adj Return")] = 1 + df_q.loc[:,("Adj Close")].pct_change()
        df_q.loc[0:1,("Adj Return")]= 1    
        #print(df_q)
        
        end = str(end)
        lower = str(start) 
        upper = str(int(lower[0:4]) + 1) + lower[4:] # One year above the lower bound
        geo_means = []        
            
        while int(lower[0:4]) < int(end[0:4]):
            #print(lower, stock)
            #print(upper, stock)
            geo_means.append(np.mean(df_q.loc[lower:upper,("Adj Return")]))
            lower =str(int(lower[0:4]) + 1) + lower[4:]
            upper = str(int(lower[0:4]) + 1) + lower[4:] 
            
        #Calculate the average over the years for an aver
        #print(geo_means)
        #Check for empty slice
        if not geo_means:
            return None, None
        
        avg = np.mean(geo_means)
        print("  "+s+": "+str(avg))

        if np.isnan(avg):
            #print("Mean Computation Stock failed: " + s)
            return None, None

        return avg, df_q.loc[:,("Adj Return")]
    except KeyError:
        pass
        #print("KeyError Stock failed: " + s)
        return None, None


def get_all_data(stock_list):
    # stock_list = list of strings containing stock names
    # df_name = name of dataframe to be stored in memory
    all_data = {}
    for stock in sorted(stock_list):
       all_data[stock] = store.get(stock)
    return all_data

            
        
parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='d', choices=['r', 'j', 'b'], help='changes default directory and file saving option. r = Rohin, j = Judah, b = Brandon', required=False, default='j')
parser.add_argument('-recov', default=True, dest='recov', required=False, help='Reform covariance matrix')
parser.add_argument('-ridge', default=True, dest='ridge', required=False, help='Perform ridge regression')
parser.add_argument('-qr', default=False, dest='qr', required=False, help='Perform QR factorization')
parser.add_argument('-sp', default=30, dest='sp', type=int, required=False, help='Shorting percentage')
parser.add_argument('-g', default=0.5, dest='g', type=float, required=False, help='Risk aversion index (gamma)')
parser.add_argument('--redownload', dest='redownload', default=False, help='regenerates the store file with data from Yahoo Finance', required=False)
parser.add_argument('--nyse', dest='nyse', action='store_true', default=True, help='disable parsing of nyse stocks', required=False)
parser.add_argument('--tsx', dest='tsx', action='store_true', default=True, help='disable parsing of nyse stocks', required=False)

args = parser.parse_args()

# Change project directory to settings for Rohin
if args.d == 'r':
    proj_dir = os.path.expanduser('~') + "/Dropbox/math441project/"
    proj_pyfl = proj_dir + "Python Files/"
    proj_savefig = proj_dir + "Figures/StockMovementRoe.png"

## ADD YOUR SETTINGS HERE BRANDON
if args.d == 'b':
    proj_dir = "D:\\Brandon\\Dropbox\\Math 441\\Project\\GitHub\\math441project\\"
    proj_pyfl = proj_dir + "Python Files\\"
    proj_savefig = proj_dir + "Figures\\StockMovementBrandon.png"


date_start = datetime.datetime(2009, 1, 1)
date_end = datetime.datetime(2016, 1, 1)


#if sys.argv[1] == "r":

# Changing directory to project folder
os.chdir(proj_pyfl)

#with open("nasdaq_symbol_list", "rb") as nasdaq_symbol_list:
#    stock_list = pickle.load(nasdaq_symbol_list)
#
#with open("sp_tsx_symbol_list", "rb") as sp_tsx_symbol_list:
#    stock_list = pickle.load(sp_tsx_symbol_list)
#
#with open("tsx_symbol_list", "rb") as nasdaq_symbol_list:
#    stock_list = pickle.load(nasdaq_symbol_list)

#Old stock list, useful for testing
stock_list = ["AAPL", "GOOG", "NFLX", "MSFT", "GE", "F", "AMZN","WMT","XOM", 
              "T", "QCOM", "MMM", "ACN", "ADBE", "BLK", "BA", "CAT", "CBS",
              "CNC", "CERN", "EA", "EXPE", "FB", "GM", "GIS", 'HOG', 'TAP',
              'IBM', 'JNJ', 'KMI', 'M', 'MCK', 'MCD', 'NKE', 'NVDA', 'PYPL',
              'PFE', 'PPG', 'PCLN', "PG", 'PWR', 'RTN', 'ROK', 'CRM', 'SHW',
              'LUV', 
              ]

failed_tickers = []
store_exists = os.path.exists(STOREFILENAME)
store = pd.HDFStore(STOREFILENAME, 'r')

all_data = get_all_data(stock_list)



data_train = {}
try:
    for stock in sorted(all_data):
        # Subsetting the matrix to desired preriod
        subsetted_data = all_data[stock].loc[date_start:date_end]
        data_train[stock] = subsetted_data
except IndexError:
    print("Stock Failed: " + stock)

del stock

#Ploting the data
def plot_data():
    for stock in sorted(data_train):
        data_train[stock].loc[:, 'Adj Close'].plot(label = stock,figsize=(8,5))
        plt.ylabel('price in USD')
    plt.legend(loc='best')
    plt.show()
    plt.savefig(proj_savefig)
    plt.close()
    return
#plot_data()


final_stock_list = []
## Obtain Returns Vector, which gives AVERAGE QUARTERLY returns for each stock
returns ={} # Returns as Hash map
mu = []    # Returns as list with no name attirbute 
ret_mat = pd.DataFrame() #monthly returns matrix
data_train_month = {}
#let's scrub out the stocks that never work so this can run a bit faster
ignore_stocks = ['AA', 'ABBV', 'AC', 'AD', 'AGI', 'AIF', 'AIM', 'ALLE', 'AQN', 'ATA', 'AVGO', 
                 'AYA', 'BB', 'BNE', 'BPY.UN.TO', 'CFG', 'CG', 'CPX.TO', 'CSRA', 'CVE', 'CWB',
                 'DG', 'DLPH', 'DSG', 'ECI', 'ECN.TO', 'EFN.TO', 'ELD', 'FB', 'FBHS', 'FCR', 
                 'FM', 'FRU', 'FSV', 'FTS', 'FTV', 'GEI.TO', 'GM', 'H', 'HBC', 'HBM', 'HCA', 
                 'HPE', 'INE', 'IVN', 'JE', 'KHC', 'KMI', 'KORS', 'KXS.TO', 'LNR', 'LUC', 
                 'LYB', 'MG', 'MJN', 'MNK', 'MNW', 'MPC', 'MX', 'NAVI', 'NEE', 'NFI', 'NLSN', 
                 'NWS', 'NWSA', 'OGC', 'OR', 'OSB', 'PLI', 'POW', 'PSK', 'PSX', 'PVG', 'PYPL', 
                 'QRVO', 'QSR', 'RNW.TO', 'RRX', 'SNC', 'SPB', 'SW', 'SYF', 'TCN.TO', 'TOG', 
                 'TOU', 'TRIP', 'UNS', 'VET', 'VRSK', 'WLTW', 'WRK', 'WSP.TO', 'XYL', 'ZTS', 'ZZZ']
avoid_stocks = ['ARX', 'DGC', 'EXE', 'PXT']
for stock in sorted(data_train):
    if stock not in ignore_stocks and args.recov==True:
        print("Recreating cov_mat, use -recov=False to use previous information")
        avg, data = annual_ret(data_train[stock], date_start, date_end, stock)
        if avg is not None and data is not None and stock not in avoid_stocks:
            returns[stock] = avg
            data_train_month[stock] = data
            ret_mat.loc[:, (stock + " Adj Return")] =  data_train_month[stock] - 1
            mu.append(avg)
            final_stock_list.append(stock)
ret_mat = ret_mat.ix[1:]
sortedkeys = sorted(returns.keys())
cov_mat = np.asmatrix(ret_mat.cov())

if args.recov==True:
    print("Total valid stocks: " + str(len(returns.keys())))
    np.savetxt(proj_pyfl + "mu.txt", ret_mat, fmt='%.5f', delimiter=", ")
    np.savetxt(proj_pyfl + "ret_mat.txt", ret_mat, fmt='%.5f', delimiter=", ")
    np.savetxt(proj_pyfl + "ret_mat_cov.txt", cov_mat, fmt='%.3f', delimiter=", ")
    np.savetxt(proj_pyfl + "mu.raw", mu)
    np.savetxt(proj_pyfl + "ret_mat.raw", ret_mat)
    np.savetxt(proj_pyfl + "cov_mat.raw", cov_mat)

## Just read locally instead of forming the covariance matrix each time
if args.recov == False:
    print("Reading previous information, use -recov=True to force recreation of cov_mat")
    mu = np.loadtxt(proj_pyfl + "mu.raw")
    ret_mat = np.loadtxt(proj_pyfl + "ret_mat.raw")
    cov_mat = np.loadtxt(proj_pyfl + "cov_mat.raw")
    print("Total valid stocks: " + str(len(cov_mat)))

## Debug code
print("\nLARGE (CO)VARIANCES (>1)")
for i in np.nditer(cov_mat, op_flags=['readwrite']):
    if i > 1:
        print("  Large value of "+str(i)+" at:")
        print("  ", np.where(cov_mat.diagonal()== i))

## MATRIX RANK BY SVD FACTORIZATION
print("\nSVD FACTORIZATION")
tol = 1e-05
svd_mat = np.linalg.svd(cov_mat, compute_uv=0)
rank = sum(np.where(svd_mat>tol,1,0))
print("  Rank of covariance matrix: ", rank)



## QR FACTORIZATION
if args.qr == True:
    print("\nFinding independent columns by QR factorization")
    tol = 1e-05
    Q,R = np.linalg.qr(cov_mat)
    icol = np.where(np.abs(R.diagonal()) > tol)[1]
    print("Found "+str(len(icol))+" independent columns")
    print("\nForming reduced return matrix and recreating covariance matrix")
    #iret = ret_mat.ix[:,icol]
    #icov = np.asmatrix(iret.cov())
    cov_mat = np.asmatrix((cov_mat[:,icol])[icol,:])
    mu = [mu[i] for i in icol]


## RIDGE REGRESSION (NOT REALLY WHAT ITS CALLED BUT WHATEVER)
if args.ridge == True:
    print("\nRIDGE REGRESSION")
    det = np.linalg.det(cov_mat)
    i = 0;
    lam = 0.27511
    lstep = 0.000001
    while det <= 0:
        i+=1
        lam += lstep
        it = np.nditer(cov_mat, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            if it.multi_index[0] == it.multi_index[1]:
                it[0] = it[0] + lam
            it.iternext()
        det = np.linalg.det(cov_mat)
    print("  Lambda: ", lam)
    print("  Iterations: ", i)
    print("  Determinant: "  + str(det))


## FIND SUBSTITUTABLE STOCKS
varv = np.diagonal(cov_mat)
corr = cov_mat / varv
acorr = corr; np.fill_diagonal(acorr,0)
alpha = 0.00001
print("\nSUBSTITUTABLE STOCKS FOR ALPHA OF", alpha)
bcorr = abs(acorr-1)<alpha
for i in range(len(bcorr)):
    for j in range(len(bcorr)):
        if bcorr[i,j]:
            print("Stock "+stock_list[i]+" is substitutable for "+stock_list[j])

diag = np.diagonal(cov_mat);
for i in diag:
    if abs(i) < 0.0001:
        print(i)
        print("INDEX: " + str(i.index));

#print("Elusive Covariance Matrix!")
#print(cov_mat)
np.savetxt(proj_pyfl + "covmatr.txt", cov_mat, fmt='%.5f', delimiter=", ")
#cov_mat = cov_mat.T.dot(cov_mat)

print("\nFUN FACTS ABOUT THE COVARIANCE MATRIX")
print("  Trace: ", cov_mat.trace())
print("  Determinant: " + str(np.linalg.det(cov_mat)))
print("  Condition Number: " + str(np.linalg.cond(cov_mat)))

#diag = cov_mat.diagonal();
#print("DIAGONAL: "  + str(diag))

print("\nFORMING OBJECTIVE")
mu = np.array(mu) # Must be converted to array before vector operations can be performed
w = cvx.Variable(len(mu))
buy = cvx.Variable(len(mu)) #Amount Bought vector
sell = cvx.Variable(len(mu)) #Amount Sold vector
t_cost = np.asarray([0.02 for x in mu]) # Transaction Cost fixed at 2%
init_alc_eql =  cvx.Parameter(len(mu))
init_alc_eql.value = np.full(len(mu), 1/len(mu))
init_alc_rnd = np.random.randint(1, 10, len(mu))
init_alc_rnd = init_alc_rnd/sum(init_alc_rnd)
init_alc_zro = np.zeros(len(mu))
ret = mu.T*w 
print("  Using gamma of " +str(args.g))
gamma = cvx.Parameter(sign='positive')
#risk = w.T*cov_mat*w
risk = cvx.quad_form(w, cov_mat)
obj = cvx.Maximize(ret - (args.g/2)*risk)
# setting the amount to leverage from other assets
print("  Using shorting percentage of " +str(args.sp))
abs_constraint = 1 + (args.sp/50)
#constraints = [cvx.sum_entries(w) == 1, w >= 0] #Budget
#constraints = [cvx.sum_entries(w) == 1, #Budget
#               cvx.sum_entries(cvx.abs(w)) <= abs_constraint]
constraints = [cvx.sum_entries(w) ==1,#+ t_cost.T*buy+ t_cost.T*sell == 1, #Budget
               cvx.norm(w, 1) <= abs_constraint,
               w == init_alc_eql + buy - sell, # Inventory
               #cvx.norm1(w) <= abs_constraint,
               buy >= 0,
               sell >= 0]
prob = cvx.Problem(obj, constraints)
gamma.value = 0.1
prob.solve(verbose = False)
rel_total = 0;
abs_total = 0;
res = np.round(w.value, 5)
print("\nOPTIMAL PORTFOLIO:")
for stock, weight in zip(sortedkeys, res):
    if abs(weight) > 0.0001:
        print("  " +stock+ ": " + str(weight[0]))
        rel_total += weight[0]
        abs_total += abs(weight[0])
print("\nOPTIMAL VALUE: " + str(prob.value))
del stock
print("  Relative sum of weights: " + str(rel_total))
print("  Shorting percentage: " +str(args.sp)+ "%")
print("  Absolute sum of weights: " + str(abs_total))


SAMPLES = 1000
risk_data = np.zeros(SAMPLES)
ret_data = np.zeros(SAMPLES)
gamma_vals = np.logspace(0, 4, num=SAMPLES)
np.logspace

# Compute trade-off curve.
def trade_off_curve():
    for i in range(SAMPLES):
        gamma.value = gamma_vals[i]
        prob.solve()
        risk_data[i] = cvx.sqrt(risk).value
        ret_data[i] = ret.value
    return

#trade_off_curve()

# Plot long only trade-off curve.
def long_trade_off_curve():
    n = len(mu)
    markers_on = [29, 40]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(risk_data, ret_data, 'g-')
    for marker in markers_on:
        plt.plot(risk_data[marker], ret_data[marker], 'bs')
        #ax.annotate(r"$\gamma = %.2f$" % gamma_vals[marker], xy=(risk_data[marker]+.08, ret_data[marker]-.03))
    for i in range(n):
        plt.plot(cvx.sqrt(cov_mat[i,i]).value, mu[i], 'ro')
    plt.xlabel('Standard deviation')
    plt.ylabel('Return')
    plt.show()
    return

store.close()
#long_trade_off_curve()
#plt.close('all')

#Determine the way to standardize our analyze
# 6 Gamma values
#Edge cases
#Conservative portfolio, average portfolio, aggressive portfolio
#Capital preservation, growth, aggressive growth

#print("optimal solution", w.value)
## Check that the weights are legitimate weights
#print(sum(w.value))


def printn(n):
    if n > 0 :
        print("foo")
        printn(n-1)
    else:
        return
        
        
for i in range(0,10):
    outtersum = []
    innersum = []
    for k in range(1,10-i):
        innersum.ape
        10-(n+k)
    sum += (10-n)
import numpy
import math   
    

ret = 0
for n in range(0, 9):
    for m in range(0, 10-(n+1)):
        inner = 10-(n+m)
        print(inner)
        ret += inner
print(ret)
    
    
    
    
ret = 0     
for n in range(0,10):
    ret += (10-n+1)*(10-n)/2
    print(ret)

    
    
    
    
    