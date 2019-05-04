import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Testing the Stationarity


def test_stationarity(df, window_rolling_mean, window_rolling_stddev):
    # Determing rolling statistics
    rolmean = df.rolling(window=window_rolling_mean, center=False).mean()

    rolstd = df.rolling(window=window_rolling_stddev, center=False).std()

    # Plot rolling statistics:
    orig = plt.plot(df, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    # drop NaN values
    df.dropna(inplace=True)

    # Perform Dickey Fuller test
    result = adfuller(df)
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
