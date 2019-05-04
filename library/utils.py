import matplotlib.pyplot as plot
from statsmodels.tsa.stattools import adfuller

# Testing the Stationarity


def test_stationarity(x):
    # Determing rolling statistics
    rolmean = x.rolling(window=22, center=False).mean()

    rolstd = x.rolling(window=12, center=False).std()

    # Plot rolling statistics:
    orig = plot.plot(x, color='blue', label='Original')
    mean = plot.plot(rolmean, color='red', label='Rolling Mean')
    std = plot.plot(rolstd, color='black', label='Rolling Std')
    plot.legend(loc='best')
    plot.title('Rolling Mean & Standard Deviation')
    plot.show(block=False)

    # Perform Dickey Fuller test
    result = adfuller(x)
    print('ADF Stastistic: %f' % result[0])
    print('p-value: %f' % result[1])
    pvalue = result[1]
    for key, value in result[4].items():
        if result[0] > value:
            print("The graph is non stationery")
            break
        else:
            print("The graph is stationery")
            break
    print('Critical values:')
    for key, value in result[4].items():
        print('\t%s: %.3f ' % (key, value))
