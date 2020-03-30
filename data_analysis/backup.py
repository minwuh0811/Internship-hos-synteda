mod = SARIMAX(ts,order=(2, 1, 2),seasonal_order=(2, 2, 0, 12),enforce_stationarity=False,enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))
plt.show()

pred = results.get_prediction(start=pd.to_datetime('2019-01-15'), dynamic=False)
print(pred.predicted_mean)
pred_ci = pred.conf_int()
ax = ts['2011':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
plt.legend()
plt.show()

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





plt.show()

plt.show()



