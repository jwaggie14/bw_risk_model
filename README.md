# bw_risk_model
Demonstration Risk Model

Go to <demonstration-bcm-model.herokuapp.com> to see a dashboard of results.

The dashboard code is in app.py.

The code to generate the fund analysis is found in fund_analysis.py. To generate the pickle files, uncomment the second to last 5 lines.
To generate the CSV files, uncomment the last 5 lines.

Return based loadings are calculated with a Kalman filter. The holdings based loadings are calculated with OLS and aggregated at the fund level by weight.

The data is sourced from Yahoo Finance, Ken French's website, and Thomson Reuters.
