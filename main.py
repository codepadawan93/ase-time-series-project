import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from statsmodels.tsa.arima_model import ARIMA
from library.utils import test_stationarity


table = pd.read_csv("resources/bitcoinData.csv", index_col=0)
# Reading the input data from csv file

# Plotting the time series
data = table['Close']
Date1 = table['Date']
train1 = table[['Date', 'Close']]
# Setting the Date as Index
train2 = train1.set_index('Date')
train2.sort_index(inplace=True)
print(type(train2))
print(train2.head())
plot.plot(train2)
plot.xlabel('Date', fontsize=12)
plot.ylabel('Price in USD', fontsize=12)
plot.title("Closing price distribution of bitcoin", fontsize=15)
plot.show()

# Log Transforming the series
ts = train2['Close']
test_stationarity(ts)

# Remove trend and seasonality with differencing
ts_log = np.log(ts)
plot.plot(ts_log, color="green")
plot.show()

test_stationarity(ts_log)

ts_log_diff = ts_log - ts_log.shift()
plot.plot(ts_log_diff)
plot.show()

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)

# Auto Regressive model
# follow lag
model = ARIMA(ts_log, order=(1, 1, 0))
results_ARIMA = model.fit(disp=-1)
plot.plot(ts_log_diff)
plot.plot(results_ARIMA.fittedvalues, color='red')
plot.title('RSS: %.7f' % sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
plot.show()

# Moving Average Model
# follow error
model = ARIMA(ts_log, order=(0, 1, 1))
results_MA = model.fit(disp=-1)
plot.plot(ts_log_diff)
plot.plot(results_MA.fittedvalues, color='red')
plot.title('RSS: %.7f' % sum((results_MA.fittedvalues-ts_log_diff)**2))
plot.show()

# Auto Regressive Integrated Moving Average Model
model = ARIMA(ts_log, order=(2, 1, 0))
results_ARIMA = model.fit(disp=-1)
plot.plot(ts_log_diff)
plot.plot(results_ARIMA.fittedvalues, color='red')
plot.title('RSS: %.7f' % sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
plot.show()


size = int(len(ts_log) - 100)
# Divide into train and test
train_arima, test_arima = ts_log[0:size], ts_log[size:len(ts_log)]
history = [x for x in train_arima]
predictions = list()
originals = list()
error_list = list()

print('Printing Predicted vs Expected Values...')
print('\n')
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
plot.figure(figsize=(8, 6))
test_day = [t for t in range(len(test_arima))]
labels = {'Orginal', 'Predicted'}
plot.plot(test_day, predictions, color='green')
plot.plot(test_day, originals, color='orange')
plot.title('Expected Vs Predicted Views Forecasting')
plot.xlabel('Day')
plot.ylabel('Closing Price')
plot.legend(labels)
plot.show()
