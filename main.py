import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from library.utils import test_stationarity, rss

import warnings
from numpy.linalg.linalg import LinAlgError
warnings.filterwarnings("ignore")

# 365 days - calculate at the level of one year
DAYS = 365
# forecast for 1000 days
FORECAST_DAYS = 1000

# Read the input data from csv file, get around the weird date format,
# set the index and sort by time
df = pd.read_csv("resources/CrudeOilPricesDaily.csv")
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
df.set_index(['Date'], inplace=True)
df.sort_index(inplace=True)
print(df.head(10))
print(type(df.index))

# Plot the time series
plt.plot(df, color='blue')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price in USD/barrel', fontsize=12)
plt.title('Closing price of crude oil', fontsize=15)
plt.show()

# Begin transforming the series to make it stationary
# Get the values only as a pandas series and apply dickie fuller
test_stationarity(df, DAYS)

# Log transform the series, plot and apply again
ts_log = np.log(df)
plt.plot(ts_log, label='Log-transformed', color='green')
plt.legend(loc='best')
plt.show()
ts_log.dropna(inplace=True)
test_stationarity(ts_log, DAYS)

# Remove trend and seasonality with differencing, plot, drop NaN values
#  and apply again
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff, label='Difference', color='green')
plt.legend(loc='best')
plt.show()
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff, DAYS)

# Decompose data into components
decomposition = seasonal_decompose(ts_log, model='multiplicative', freq=DAYS)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.subplot(411)
plt.plot(ts_log, label='Original', color='blue')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend', color='blue')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality', color='blue')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals', color='blue')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Test stationarity of residuals (noise) to find
plt.plot(residual, label='Residuals (noise)', color='blue')
plt.legend(loc='best')
plt.show()
test_stationarity(residual, DAYS)
plt.show()

# Plot ACF and PACF to find p, d, q - smoothen using rolling mean
log_rollmean = ts_log.rolling(window=DAYS, center=False).mean()
log_rollmean.dropna(inplace=True)
lag_acf = acf(log_rollmean, nlags=2600)
lag_pacf = pacf(log_rollmean, nlags=20, method='ols')
print(lag_acf)
print(lag_pacf)

# Plot ACF:
plt.subplot(121)
plt.plot(lag_acf, color='blue')
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(
    y=-1.96/np.sqrt(len(ts_log)), linestyle='--', color='gray')
plt.axhline(
    y=1.96/np.sqrt(len(ts_log)), linestyle='--', color='gray')
plt.title('Autocorrelation Graph')

# Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf, color='blue')
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Graph')
plt.tight_layout()
plt.show()

# Auto Regressive model
model = ARIMA(ts_log['Closing Value'], order=(1, 1, 0))
results_AR = model.fit(disp=-1)
print(results_AR.fittedvalues - ts_log_diff['Closing Value'])
plt.plot(ts_log_diff, color='blue')
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.7f' %
          rss(results_AR.fittedvalues, ts_log_diff['Closing Value']))
plt.show()

# Moving Average Model
model = ARIMA(ts_log['Closing Value'], order=(0, 1, 1))
results_MA = model.fit(disp=-1)
plt.plot(ts_log_diff, color='blue')
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.7f' %
          rss(results_MA.fittedvalues, ts_log_diff['Closing Value']))
plt.show()

# Auto Regressive Integrated Moving Average Model
model = ARIMA(ts_log['Closing Value'], order=(2, 1, 0))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff, color='blue')
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.7f' %
          rss(results_ARIMA.fittedvalues, ts_log_diff['Closing Value']))
plt.show()

# min_rss = 1000000
# params = (0, 0, 0)
# for i in range(0, 100):
#     j = k = 1
#     print(i, j, k)
#     try:
#         model = ARIMA(ts_log['Closing Value'], order=(i, j, k))
#         results_ARIMA = model.fit(disp=-1)
#         _rss = rss(results_ARIMA.fittedvalues,
#                    ts_log_diff['Closing Value'])
#         print('RSS is', _rss)
#         if(_rss < min_rss):
#             min_rss = _rss
#             params = (i, j, k)
#             print('current min RSS', min_rss)
#     except ValueError:
#         print('Value Error occurred')
#         continue
#     except LinAlgError:
#         print('LinAlg Error occurred')
#         continue
# print('Best fit: ', params)
# print('RSS: ', min_rss)

# we can now make predictions
# Divide into train and test datasets
size = int(len(ts_log) - 50)
train_arima, test_arima = ts_log['Closing Value'][0:size], ts_log['Closing Value'][size:len(
    ts_log)]
history = [x for x in train_arima]
predictions = list()
originals = list()
error_list = list()

print('Predicted vs Expected Values\n')
# We go over each value in the test set and then apply ARIMA model and calculate the predicted value.
# We have the expected value in the test set therefore we calculate the error between predicted and expected value
for t in range(len(test_arima)):
    model = ARIMA(history, order=(2, 1, 0))
    model_fit = model.fit(disp=-1)

    output = model_fit.forecast()

    pred_value = output[0]

    original_value = test_arima[t]
    history.append(original_value)

    pred_value = np.exp(pred_value)

    original_value = np.exp(original_value)

    # Calculating the error
    error = ((abs(pred_value - original_value)) / original_value) * 100
    error_list.append(error)
    print('predicted = %f,   expected = %f,   error = %f ' %
          (pred_value, original_value, error), '%')

    predictions.append(float(pred_value))
    originals.append(float(original_value))

# After iterating over whole test set the overall mean error is calculated.
print('\n Mean Error in Predicting Test Case Articles : %f ' %
      (sum(error_list) / float(len(error_list))), '%')
plt.figure(figsize=(8, 6))
test_day = [t for t in range(len(test_arima))]
plt.plot(test_day, predictions, color='blue')
plt.plot(test_day, originals, color='orange')
plt.title('Expected Vs Predicted Views Forecasting')
plt.xlabel('Day')
plt.ylabel('Closing Value')
plt.legend({'Original', 'Predicted'})
plt.show()

# Now we can forecast values
print(ts_log.head(10))
results_ARIMA.plot_predict(1, len(ts_log['Closing Value']) + 1000)
plt.show()

x_vals = [x for x in range(len(ts_log['Closing Value']) + 1000)]
y_vals = [np.exp(x) for x in ts_log['Closing Value']]
y_vals2 = [np.nan for x in ts_log['Closing Value']]
for x in results_ARIMA.forecast(1000)[0]:
    y_vals.append(np.nan)
    y_vals2.append(np.exp(x))
plt.plot(x_vals, y_vals, label='Original', color='orange')
plt.plot(x_vals, y_vals2, label='Predicted', color='blue')
plt.show()
