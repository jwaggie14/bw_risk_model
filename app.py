# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:29:12 2019

@author: jlwhi
"""
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

bwmf = {'BRAGX':'Aggressive Investors 1',
       'BRSGX':'Small-Cap Growth',
       'BRSVX':'Small-Cap Value',
       'BRUSX':'Ultra-Small Company',
       'BRLIX':'Blue Chip',
       'BOSVX':'Omni Small-Cap Value',
       'BOTSX':'Omni Tax-Managed Small-Cap Value',
       'BRISX':'Ultra-Small Company Market'}

dd1_options = []

bwmf2 = {'BRAGX':'Aggressive Investors 1',
       'BRSVX':'Small-Cap Value',
       'BRUSX':'Ultra-Small Company',
       'BRLIX':'Blue Chip',
       'BOSVX':'Omni Small-Cap Value',
       'BOTSX':'Omni Tax-Managed Small-Cap Value',
       }

for f in bwmf2:
    dd1_options.append({'label': bwmf2[f], 'value': f})


prior = pd.Series(data=[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],index=['Alpha','Mkt-RF','SMB','HML','WML','CMA','RMW','STR'])

fmodels = {'full':['Alpha','Mkt-RF','SMB','HML','CMA','RMW','WML','STR'],
          'ff3':['Alpha','Mkt-RF','SMB','HML'],
          'ff5':['Alpha','Mkt-RF','SMB','HML','CMA','RMW'],
          'carhart':['Alpha','Mkt-RF','SMB','HML','WML'],
          'capm':['Alpha','Mkt-RF']}

dd2_options = [
        {'label': 'Extended (Fama French 5 + Momentum + Short Term Reversals)', 'value': 'full'},
        {'label': 'Fama French 3 Factor', 'value': 'ff3'},
        {'label': 'Fama French 5 Factor', 'value': 'ff5'},
        {'label': 'Carhart 4 Factor', 'value': 'carhart'},
        {'label': 'CAPM', 'value': 'capm'},
        ]


bwmfs = list(bwmf.keys())
# yahoo did not return values for these funds
bwmfs.remove('BRSGX')
bwmfs.remove('BRISX')

models = list(fmodels.keys())

fund_exposures = pd.read_csv('fe.csv')
fund_exposures['Date'] = pd.to_datetime(fund_exposures['Date'])
hold_loadings = pd.read_csv('hl.csv')
rets = pd.read_csv('rets.csv')
rets['Date'] = pd.to_datetime(rets['Date'])
rets = rets.set_index('Date')

app.layout = html.Div(children=[
    html.H1(children='BCM Demonstration Risk Model'),

    dcc.Markdown('''
This app is intended to be a high-level demonstration of a return and holdings based risk model. The time-series loadings are estimated using a Kalman filter. The Kalman filter performed comparably to a 2 year rolling OLS esimate, but has the advantage of not requiring a fixed window. The drawback to the kalman filter is it requires a 'burn-in' period if the specified priors are significantly different from the 'true' loadings. The holdings exposures were estimated with OLS to demonstrate a different technique and to show the consistency with the results of the Kalman filter estimates.
        
The return data was sourced from Yahoo Finance. Some of the Bridgeway funds did not return prices, so they are not included below. Further, some of the holdings symbols did not map to a Yahoo Finance symbol, particularly the options, and were excluded also. The small/value funds were particularly impacted. As such, the holding loadings are unreliable.

The factors were pulled from Ken French's data library and the holding information was collected from Thomson Reuters Eikon.

### How To Use

You can select different BCM funds and several demo factor models using the dropdowns below. The graphs will update automatically on selection.
    '''),
             
    html.Label('Fund Selector'),
    dcc.Dropdown(id='dd1',
        options=dd1_options,
        value='BRAGX'
    ),
    html.Div(children="<br>"),
    
    html.Label('Model Selector'),
    
    dcc.Dropdown(id='dd2',
        options=dd2_options,
        value='full'
    ),

    html.Div(id='fig1'),
    html.Div(id='fig2'),
    html.Div(id='fig3')
    
    
])

@app.callback(
    Output(component_id='fig1', component_property='children'),
    [Input(component_id='dd1', component_property='value'),
     Input(component_id='dd2', component_property='value')]
)
def plot_exposure(fund='BRAGX',model='full'):
    start_date = '1/1/1990'
    end_date = '1/1/2020'
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    
    data = []
    
    x = fund_exposures[(fund_exposures['fund']==fund)&(fund_exposures['model']==model)][['Date']]
    x = x[(x['Date'] >= sd) & (x['Date'] <= ed)]
    
    for mod in fmodels[model][1:]:
#        print(x)
        
        y = fund_exposures[(fund_exposures['fund']==fund)&(fund_exposures['model']==model)][mod].loc[x.index]
#        print(y)
        data.append({'x': x.iloc[:,0].to_list(), 'y': y.to_list(), 'type': 'line', 'name': mod})

    gr = dcc.Graph(
            id='return-based-loadings',
            figure={
                    'data': data,
                    'layout': {
                            'title': f'{fund} Factor Loadings'
                            }
                    })
    return [gr]
    

@app.callback(
    Output(component_id='fig2', component_property='children'),
    [Input(component_id='dd1', component_property='value'),
     Input(component_id='dd2', component_property='value')]
)
def estimate_rolling(fund='BRAGX',model='full'):
    fe = fund_exposures.copy()
    
    temp = hold_loadings[(hold_loadings['fund']==fund) & (hold_loadings['model']==model)]
    temp['Alpha'] *= 252

    
    fes = fe[(fe['fund']==fund)&(fe['model']==model)].iloc[-1][fmodels[model]]
    fes['Alpha'] *= 252
    hold = temp[fmodels[model]].multiply(temp['Weight'],axis=0).sum().values
    
    labels = [i for i in fmodels[model]]
    labels[0] = 'Annualized Alpha'
    
#    x = list(range(len(fmodels[model])))
    x = fmodels[model]
    data2 = []
    data2.append({'x':x, 'y':list(fes.values), 'type':'bar', 'name':'Kalman Filter Loading Estimates'})
    data2.append({'x':x, 'y':list(hold), 'type':'bar', 'name':'Aggregated Holdings Loading Estimates'})
    
    
    fig2 = dcc.Graph(id='comparison-loadings',
                     figure={
                             'data': data2,
                             'layout': {'title':'Comparison of Holdings vs Return Based Models'}}
            )
    
    stacks = temp[fmodels[model]].multiply(temp['Weight'],axis=0)
    stacks = stacks.merge(temp['Name'],left_index=True,right_index=True)
    stacks = stacks.values
    
    data = []
    
    
    
    
    for row in stacks: 
        data.append(go.Bar(
                x=x, y=row[:-1], name=row[-1]))

    layout = go.Layout(
            barmode='stack',
            title=f"{fund} Loadings Based on 6/30 Holdings",
            xaxis=dict(tickvals=fmodels[model]),
            showlegend=False)

    fig = dcc.Graph(id='holdings-based-loadings',figure=go.Figure(data=data, layout=layout))

    
    return [fig, fig2]


def residual_return(fund='BRAGX',model='full'):
    fe = fund_exposures.set_index('Date',drop=False).copy()
    b = fe[(fe['fund']==fund)&(fe['model']==model)][fmodels[model][1:]]
    
    a = rets[fmodels[model][1:]].loc[b.index]
    c = a.multiply(b)
    d = rets[fund].pct_change().dropna() - rets['RF']
    d.name = f'{fund}-RF'
    
    residual = (d - c.sum(axis=1)).dropna()
    return d, c, residual, rets['RF'].loc[residual.index]


@app.callback(
    Output(component_id='fig3', component_property='children'),
    [Input(component_id='dd1', component_property='value'),
     Input(component_id='dd2', component_property='value')]
)
def return_attribution(fund='BRAGX', model='full'):
    #a excess, b is common return, c is specific, d is risk free
    a, b, c, d = residual_return(fund, model)

    
    a = (b.sum(axis=1)+c +d).cumsum() #total
    b = b.sum(axis=1).cumsum() # common
    c = c.cumsum() #specific
    d = d.cumsum() #rf

    x = a.index.to_list()
    data = []
    data.append({'x':x, 'y':a.to_list(), 'type':'line', 'name':'Total Return'})
    data.append({'x':x, 'y':b.to_list(), 'type':'line', 'name':'Common Return'})
    data.append({'x':x, 'y':c.to_list(), 'type':'line', 'name':'Specific Return'})
    data.append({'x':x, 'y':d.to_list(), 'type':'line', 'name':'Risk Free Return'})
    
    fig = dcc.Graph(id='return-attribution',
                     figure={
                             'data': data,
                             'layout': {'title':'Return Attribution Using Kalman Loadings (Log Returns)'}}
            )
    return [fig]

if __name__ == '__main__':
    app.run_server(debug=True)