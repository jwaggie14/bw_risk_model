# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 22:24:25 2019

@author: jlwhi
"""

import pandas as pd
import numpy as np
import pandas_datareader.data as web
import statsmodels.api as sm
from pykalman import KalmanFilter
from numpy.lib.stride_tricks import as_strided as stride
import warnings


pd.set_option('mode.chained_assignment', None)

### Functions ###




def time_varying_factors(rets, fund, model='full'):
    """Returns dynamic factor loadings calculated with a Kalman Filter.
    
    Parameters
    ----------
    rets : dataframe with the Factor and Fund returns as columns with date index.
    fund : string specifying which fund - e.g. 'BRAGX'
    model : string specifying which model. Can be 'full','ff3','ff5','carhart', or 'capm'
    
    Returns
    -------
    kalman_factors : dataframe with time series of factor loading estimates
    
    """
    
    
    # define the factor model   
    if model == 'full':
        factors = ['Mkt-RF','SMB','HML','RMW','CMA','WML','STR'] 
    elif model == 'ff3':
        factors = ['Mkt-RF','SMB','HML']
    elif model == 'ff5':
        factors = ['Mkt-RF','SMB','HML','RMW','CMA']
    elif model == 'carhart':
        factors = ['Mkt-RF','SMB','HML','WML']
    elif model == 'capm':
        factors = ['Mkt-RF']
    
    #reduce rets to the relevant columns
    temp_df = rets[factors + ['RF',fund]]
    
    #calculate simple returns for the fund
    temp_df['returns'] = temp_df[fund].pct_change()
    temp_df = temp_df.dropna()

    
    x_r = temp_df[factors]
    x_r = sm.add_constant(x_r)

    y_r = temp_df['returns'] - temp_df['RF']
    
    #originally tested a rolling OLS regression. Kalman outperformed all & performed comparably to 2y rolling window
#     df = temp_df.merge(roll(temp_df, 1000).apply(rolling_betas),how='left',left_index=True,right_index=True)
#     coefs = df[df.columns[-8:]].shift(999)
    
    
    #Prep arguments for kalman filter
    dimensions = len(factors) + 1

    #Define the Prior's for each Fund and Model combination.     
    if model == 'capm':
        ism = [0,1]
    else:
        if fund in ['BOTSX','BRSVX','BOSVX']:
            ism = [0,1,1,.5] + [0]*(dimensions-4)
        elif fund == 'BRUSX':
            ism = [0,1,1] + [0]*(dimensions-3)
        else:
            ism = [0,1] + [0]*(dimensions-2)
    
    
    ### Define the model ###
    
    delta_r = 1e-2 # determines how quickly the kalman filter reacts to new information
    trans_cov_r = delta_r / (1 - delta_r) * np.eye(dimensions) # How much random walk wiggles
    obs_mat_r = np.expand_dims(x_r, axis=1)
    kf_r = KalmanFilter(n_dim_obs=1, n_dim_state=dimensions, # y_r is 1-dimensional, full model, (alpha, beta, SMB, HML, RMW, CMA, WML, STR), is 7-dimensional
                      initial_state_mean=ism, #List of our "Priors" e.g. beliefs about the expected loadings.
                      initial_state_covariance=np.ones((dimensions, dimensions)), #using an uninformed prior covariance matrix.
                      transition_matrices=np.eye(dimensions),
                      observation_matrices=obs_mat_r,
                      observation_covariance=.01,
                      transition_covariance=trans_cov_r)

    ### Fit the model ###
    results = kf_r.filter(y_r.values) 

    ### Format results for output ###
    kalman_factors = pd.DataFrame(results[0], columns=['Alpha'] + temp_df.columns[:(dimensions-1)].tolist(), index=temp_df.index)
    kalman_factors['fund'] = fund
    kalman_factors['model'] = model
    kalman_factors = kalman_factors.reset_index()
    
    return kalman_factors

def estimate_params(x, model='ff3',roll=500):
    """Use with pd.DataFrame.apply() to estimate loadings for each stock in a dataframe.
    
    Parameters
    ----------
    x : dataframe with the Factor and Fund returns as columns with date index.
    model : string specifying which model. Can be 'full','ff3','ff5','carhart', or 'capm'
    
    Returns
    -------
    loadings : series that maps to columns in the applied dataframe.
    
    """
    #specify the model
    factors = fmodels[model][1:]
    
    if x in stocks.columns: #if we have data for the stock, apply the model
        Y = (stocks[x].pct_change() - rets['RF']).dropna()
        X = rets[factors].loc[Y.index]
        X = sm.add_constant(X)
        X = X.iloc[-roll:]
        Y = Y.iloc[-roll:]
        mod = sm.OLS(Y,X)
        res = mod.fit()
        p = res.params
        p.index = fmodels[model]
        return p
    else:
        #if the stock is missing data, returns a naive estimate of loadings. e.g. Beta 1, all other factors 0
        return prior[fmodels[model]]



def strip(x):
    """ Helper function to clean the Stock symbols """
    if type(x) == str:
#         print(x)
        try: x = x[:x.index('.')]
        except: x = x
        return x
    else:
        return 'BIL'

def roll(df, w, **kwargs):
    """ Helper function to apply rolling calculation to multiple columns """
    
    v = df.values
    d0, d1 = v.shape
    s0, s1 = v.strides

    a = stride(v, (d0 - (w - 1), w, d1), (s0, s0, s1))

    rolled_df = pd.concat({
        row: pd.DataFrame(values, columns=df.columns)
        for row, values in zip(df.index, a)
    })

    return rolled_df.groupby(level=0, **kwargs)



def rolling_betas(df):
    """helper functino to calculate rolling OLS loadings. Not used after testing"""
#     print(df.shape)
#     print(df)
    X = df[['Mkt-RF','SMB','HML','RMW','CMA','WML','STR']]
    X = sm.add_constant(X)

    Y = df['returns'] - df['RF']
    mod = sm.OLS(Y,X)
    res = mod.fit()
    t = res.params
    t.index = [f"{a}_c" for a in t.index]
    return t



###################
### Data Processing
###################
    

# Collect the Factors from French's website
ff = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily','famafrench',start='1/1/1990')
ff_m = web.DataReader('F-F_Momentum_Factor_daily','famafrench',start='1/1/1990')
ff_st = web.DataReader('F-F_ST_Reversal_Factor_daily','famafrench',start='1/1/1990')
ff_i = web.DataReader('10_Industry_Portfolios_daily','famafrench',start='1/1/1990')


ff = ff[0] #ff returns factors in multiple time intervals

ff = ff.join(ff_m[0]).join(ff_st[0]).join(ff_i[0]) #combine all the factors into one dataframe
ff /= 100 # convert returns into percentages

#ff.to_pickle('ff.pkl')

# Dict of BCM funds
bwmf = {'BRAGX':'Aggressive Investors 1',
       'BRSGX':'Small-Cap Growth',
       'BRSVX':'Small-Cap Value',
       'BRUSX':'Ultra-Small Company',
       'BRLIX':'Blue Chip',
       'BOSVX':'Omni Small-Cap Value',
       'BOTSX':'Omni Tax-Managed Small-Cap Value',
       'BRISX':'Ultra-Small Company Market'}

#define the priors used for the kalman filters
prior = pd.Series(data=[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],index=['Alpha','Mkt-RF','SMB','HML','WML','CMA','RMW','STR'])


# collect the holdings data from files
holdings = None
for key in bwmf:
    df = pd.read_excel(f'{key}.xlsx', header=3)
    df['fund'] = key
    df['symbol'] = df['RIC'].apply(strip)
    #if the df exists, join them. if it doesn't, create it
    if isinstance(holdings, type(None)):
        holdings = df
    else:
        holdings = holdings.append(df,ignore_index=True)

#list of symbols to try with Yahoo Finance. Prior testing showed that symbols with > 5 characters didn't have a match (all options)
syms = [x for x in holdings.symbol.unique().tolist() if len(x) <= 5]

#dfs.to_pickle('stocks.pkl')

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

stocks = []
i = 0
length = len(syms)

#split the symbols into bite sized chunks. Prevents total failure if one symbol fails.
for port in np.array_split(syms,length//2):
    try:
        df = web.DataReader(port,'yahoo',start='1/1/2015')['Adj Close']
        stocks.append(df)
        i += 1
        print(f'{i} out of {length//2}', end='\r')
    except:
        try:
            df = web.DataReader(port,'yahoo',start='1/1/2015')['Adj Close']
            stocks.append(df)
            i += 1
            print(f'{i} out of {length//2}', end='\r')
        except:
            print(f'failed on {i}')

#combine all of the stock returns into one df
dfs = None
for df in stocks:
    if isinstance(dfs, type(None)):
        dfs = df
    else:
        dfs = dfs.join(df)


bwmfs = list(bwmf.keys())
# yahoo did not return values for these funds
bwmfs.remove('BRSGX')
bwmfs.remove('BRISX')

#Pull return data for BCM funds
bridgeway = None
for fund in bwmfs:
    print(fund)
    df = web.DataReader(fund,'yahoo',start='1/1/1990')['Adj Close']
    df.name = fund
    if isinstance(bridgeway, type(None)):    
        bridgeway = pd.DataFrame(df)
    else:
        bridgeway = bridgeway.join(df, how='outer')

#combine fund returns with factor returns
rets = ff.join(bridgeway)
rets = rets.dropna(thresh=19)

#redefine column names
x = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'WML', 'STR', 'NoDur',
       'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm', 'Shops', 'Hlth ', 'Utils',
       'Other']
x.extend(bwmfs)
rets.columns = x

#rets.to_pickle('rets903.pkl')

#dict of factor models
fmodels = {'full':['Alpha','Mkt-RF','SMB','HML','CMA','RMW','WML','STR'],
          'ff3':['Alpha','Mkt-RF','SMB','HML'],
          'ff5':['Alpha','Mkt-RF','SMB','HML','CMA','RMW'],
          'carhart':['Alpha','Mkt-RF','SMB','HML','WML'],
          'capm':['Alpha','Mkt-RF']}


models = list(fmodels.keys())


########################################
### Calculate the Kalman Filter Loadings
########################################

fund_exposures = None
for f in bwmfs:
    for m in fmodels:
        #time varying factors function calculate the loadings
        tdf = time_varying_factors(rets,f,m)
        if isinstance(fund_exposures, type(None)):
            fund_exposures = tdf
        else:
            fund_exposures = pd.concat([fund_exposures,tdf],ignore_index=True)

#fund_exposures.to_pickle('fund_exposures903.pkl')

########################################
### Calculate the loadings for Holdings
########################################

hold_loadings = None
for mod in fmodels.keys():
    # estimate_params returns OLS loadings for stock with defined lookback (roll)
    df = holdings['symbol'].apply(lambda x: estimate_params(x, model=mod, roll=500))
    df['fund'] = holdings['fund']
    df['Weight'] = holdings['Weight']
    df['model'] = mod
    if isinstance(hold_loadings, type(None)):
        hold_loadings = df
    else:
        hold_loadings = pd.concat([hold_loadings, df])
        

#hold_loadings.to_pickle('hold_loadings.pkl')
