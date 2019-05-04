import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Testing the Stationarity of the series


def test_stationarity(x, window):
    # Determine rolling statistics
    rollmean = x.rolling(window=window, center=False).mean()
    rollstd = x.rolling(window=window, center=False).std()

    # Plot rolling statistics:
    plt.plot(x, color='blue', label='Original')
    plt.plot(rollmean, color='red', label='Rolling Mean')
    plt.plot(rollstd, color='black', label='Rolling Std deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # drop NaN values as it causes the test to crash
    x2 = x.dropna()

    # Perform Dickey Fuller test and log results
    result = adfuller(x2['Closing Value'], autolag='AIC')
    print('Dickey-Fuller Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    pvalue = result[1]
    for key, value in result[4].items():
        print('Critical value %s: %.3f ' % (key, value))
    for key, value in result[4].items():
        if result[0] > value:
            print('The dataset is non-stationary')
            break
        else:
            print('The dataset is stationary (confidence level %s)' % key)
            break
