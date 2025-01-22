import pandas as pd
import yfinance as yf
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def fetch_s_and_p_500_data(start_date, end_date):
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    data = pd.read_html(url)
    tickers = data[0]['Symbol'].tolist()
    price_data = yf.download(tickers, start=start_date, end=end_date)['Close']
    return price_data


def fill_missing_val(data):
    # Drop columns with more than 10% missing data
    if data.isnull().values.any():
        missing_fractions = data.isnull().mean().sort_values(ascending=False)
        drop_list = sorted(list(missing_fractions[missing_fractions > 0.1].index))
        data.drop(labels=drop_list, axis=1, inplace=True)
        data = data.fillna(method='ffill').fillna(method='bfill')
    return data

def compute_norm_spread(S1, S2):
    X = sm.add_constant(S1)
    model = sm.OLS(S2, X).fit()
    spread = S2 - model.predict(X)
    normalized_spread = (spread - spread.mean()) / spread.std()
    return normalized_spread

def calculate_half_life(spread):
    """
    Calculate the half-life of mean reversion for a spread.

    Parameters:
    - spread: time series of the spread.

    Returns:
    - half_life: estimated half-life of mean reversion.
    """
    spread_lag = spread.shift(1).fillna(method="bfill")
    delta_spread = spread - spread_lag
    spread_lag_const = sm.add_constant(spread_lag)
    model = sm.OLS(delta_spread, spread_lag_const)
    res = model.fit()
    gamma = res.params.iloc[1]
    half_life = -np.log(2) / gamma
    return half_life

def calculate_mean_crossings(spread, mean = 0.0):
    """
    Counts the number of times a time series crosses a specified mean value.

    :param time_series: A pandas Series or array-like object representing 
                        the time series data.
    :param mean: The mean value to check crossings against. Default is 0.0.
    :return: The total number of mean crossings in the time series.
    """

    # Current and next period values for comparison
    current_series = spread
    next_series = spread.shift(-1)  # Shift by -1 to align with the next period

    # Initialize total crossings count
    total_crossings = 0

    # Iterate through the current and next values to count crossings
    for current, next_ in zip(current_series, next_series):
        if current >= mean and next_ < mean:  # Over to under
            total_crossings += 1
        elif current < mean and next_ >= mean:  # Under to over
            total_crossings += 1
        elif current == mean:  # Exactly at the mean
            total_crossings += 1

    return total_crossings

'''
def compute_hurst_exp(spread, max_lag=100):
    lags = range(2, max_lag)
    tau = []  # Variance of the differenced series at different lags
    for lag in lags:
        # Calculate the lagged differences
        diff = spread[lag:] - spread[:-lag]
        # Compute the standard deviation (tau) at this lag
        tau.append(np.sqrt(np.std(diff)))
    # Perform a linear fit on the log-log plot
    res = np.polyfit(np.log(lags), np.log(tau), 1)
    H = res[0] * 2.0
    return H
'''
def find_cointegrated_pairs(tickers, data):
    """
    Find cointegrated pairs among a list of tickers using the Engle-Granger test
    and applying the four criteria from Sarmento and Horta.

    Parameters:
    - tickers: list of ticker symbols.
    - data: DataFrame containing price data for the tickers.

    Returns:
    - pairs: list of dictionaries containing the cointegrated pairs and their statistics.
    """
    n = len(tickers)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            ticker1 = tickers[i]
            ticker2 = tickers[j]
            if ticker1 not in data.columns:
                print('error!!')
            asset1 = data[ticker1]
            asset2 = data[ticker2]

            # Perform cointegration test
            tstat, pvalue, crit_value = coint(asset1, asset2)
            critical_value = crit_value[1]  # 0: 1%, 1: 5%, 2:10%
            if tstat < critical_value: 
                spread = compute_norm_spread(asset1, asset2)
                #hurst_exp = compute_hurst_exp(spread)
                #if hurst_exp < 0.5:
                # Calculate the half-life of mean reversion
                half_life = calculate_half_life(spread)
                # Only select pairs with half-life between 1 and 252
                if 1 < half_life < 252:
                    # Calculate mean crossings per year
                    mean_crossings_per_year = calculate_mean_crossings(spread, 0.0)
                    if mean_crossings_per_year >= 12:
                        pairs.append({
                            'pair': (ticker1, ticker2),
                            'tstat': tstat,
                            'pvalue': pvalue,
                            'critical_value_5pct': critical_value,
                            'half_life': half_life,
                            'mean_crossings_per_year': mean_crossings_per_year
                            })
    return pairs

