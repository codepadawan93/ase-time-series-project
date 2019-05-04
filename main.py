import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from library.utils import test_stationarity


# Reading the input data from csv file and get around the weird date format
df = pd.read_csv("resources/CrudeOilPricesDaily.csv")
print(df.head(10))
df['Date'] = pd.to_datetime(df['Date'])
df.set_index(df['Date'])

# Plotting the time series
data = df['Closing Value']
Date1 = df['Closing Value']
train1 = df[['Date', 'Closing Value']]

# Setting the Date as Index
train2 = train1.set_index('Date')
train2.sort_index(inplace=True)
print(type(train2))
print(train2.head())
plt.plot(train2)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price in USD', fontsize=12)
plt.title("Closing price distribution of crude oil", fontsize=15)
plt.show()

# Log Transforming the series
ts = train2['Closing Value']
test_stationarity(ts, 367, 1000)

# Remove trend and seasonality with differencing
ts_log = np.log(ts)
plt.plot(ts_log, color="green")
plt.show()

test_stationarity(ts_log, 367, 1000)

ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
plt.show()

ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff, 367, 1000)

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
