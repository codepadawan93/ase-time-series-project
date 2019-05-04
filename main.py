import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from library.utils import test_stationarity

# 365 days - calculate at the level of one year
DAYS = 365

# Read the input data from csv file, get around the weird date format,
# set the index and sort by time
df = pd.read_csv("resources/CrudeOilPricesDaily.csv")
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
df.set_index(['Date'], inplace=True)
df.sort_index(inplace=True)
print(df.head(10))
print(type(df.index))

# Plot the time series
plt.plot(df)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price in USD/barrel', fontsize=12)
plt.title('Closing price of crude oil', fontsize=15)
plt.show()

# Begin transforming the series to make it stationary
# Get the values only as a pandas series and apply dickie fuller
test_stationarity(df, DAYS)

# Log transform the series, plot and apply again
ts_log = np.log(df)
plt.plot(ts_log, label='Log-transformed')
plt.legend(loc='best')
plt.show()
ts_log.dropna(inplace=True)
test_stationarity(ts_log, DAYS)

# Remove trend and seasonality with differencing, plot, drop NaN values
#  and apply again
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff, label='Difference')
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
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# TODO:: Fix below, video was at 26.55

# Auto Regressive model
# follow lag
model = ARIMA(ts_log, order=(1, 1, 0))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.7f' % sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
plt.show()

# Moving Average Model
# follow error
model = ARIMA(ts_log, order=(0, 1, 1))
results_MA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.7f' % sum((results_MA.fittedvalues-ts_log_diff)**2))
plt.show()

# Auto Regressive Integrated Moving Average Model
model = ARIMA(ts_log, order=(2, 1, 0))
results_ARIMA = model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.7f' % sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
plt.show()

# we can now make predictions
# Divide into train and test datasets
size = int(len(ts_log) - 50)
train_arima, test_arima = ts_log[0:size], ts_log[size:len(ts_log)]
history = [x for x in train_arima]
predictions = list()
originals = list()
error_list = list()

print('Printing Predicted vs Expected Values...\n')
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
labels = {'Orginal', 'Predicted'}
plt.plot(test_day, predictions, color='green')
plt.plot(test_day, originals, color='orange')
plt.title('Expected Vs Predicted Views Forecasting')
plt.xlabel('Day')
plt.ylabel('Closing Price')
plt.legend(labels)
plt.show()
