import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics import tsaplots
import itertools
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv("raw_data.csv", sep=',' ,skiprows=1)
data.columns = ["time", "value", "2","3","4"]
data=data.drop(["2","3","4"], axis=1)
data['time'] = pd.to_datetime(data['time'])
data.set_index('time', inplace=True)
ts = data['value']


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    # Plot rolling statistics:
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    # Perform Dickey-Fuller test:
    print( 'Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
test_stationarity(ts)

ts_log=np.log(ts)
ts_log_mean=ts_log.rolling(12).mean()
ts_log_wmean= ts_log.ewm(halflife=12,min_periods=0,adjust=True,ignore_na=False).mean()
diff=np.subtract(ts_log, ts_log.shift())
diff.dropna(inplace=True)
#test_stationarity(diff)

diff1 = ts.diff(1)
diff1=np.subtract(diff1, diff1.shift())
diff1.dropna(inplace=True)
print(diff1)
test_stationarity(diff1)
plt.plot(diff1)
plt.show()

diff1.dropna(inplace=True)
fig = plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig = tsaplots.plot_acf(diff1,lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = tsaplots.plot_pacf(diff1,lags=40,ax=ax2)
fig.show()

lag_acf=acf(diff,nlags=20)
lag_pacf = pacf(diff,nlags=20, method='ols')

plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(diff)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(diff)), linestyle='--', color='gray')
plt.title('Patial Autocorrelation Function')
plt.tight_layout()
plt.show()

"""
p = q =d = range(0, 5)
pdq = list(itertools.product(p, 1, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(ts, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

p = q =d = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 13) for x in list(itertools.product(p, d, q))]
for param in pdq:
    for param_seasonal in seasonal_pdq:
        mod = SARIMAX(ts, order=param, seasonal_order=param_seasonal, enforce_stationarity=False,
                  enforce_invertibility=False)

        results = mod.fit()
        pred = results.get_prediction(start=pd.to_datetime('2019-01-15'), dynamic=False)
        y_forecasted = pred.predicted_mean
        y_truth = ts['2019-01-15':]
        mse = ((y_forecasted - y_truth) ** 2).mean()
        print('ARIMA{}x{}12 '.format(param, param_seasonal))
        print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
        print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
"""
mod = SARIMAX(ts,order=(11, 2, 1),seasonal_order=(2, 2, 0, 12),enforce_stationarity=False,enforce_invertibility=False)
results = mod.fit()
print('AIC:{} BIC:{} HQ:{}'.format (results.aic, results.bic, results.hqic))
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))
plt.show()
plt.subplot(121)
pred = results.get_prediction(start=pd.to_datetime('2019-04-15'), dynamic=False)
print(pred.predicted_mean)
pred_ci = pred.conf_int()
ax = ts['2011':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
plt.legend()
plt.subplot(122)
y_forecasted = pred.predicted_mean
y_truth = ts['2019-01-15':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

fcast_index = pd.to_datetime(['2020-04-15', '2020-05-15', '2020-06-15', '2020-07-15', '2020-08-15', '2020-09-15', '2020-10-15', '2020-11-15', '2020-12-15', '2021-01-15', '2021-02-15', '2021-03-15'])
pred_uc = results.get_forecast(steps=12, index=fcast_index)
print(pred_uc.predicted_mean)
pred_ci = pred_uc.conf_int()
ax = ts.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Year')
ax.set_ylabel('Value')
plt.legend()
plt.show()


