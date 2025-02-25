# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#
aapl_file = 'AAPL.csv'
amzn_file = 'AMZN.csv'
googl_file = 'GOOGL.csv'
msft_file = 'MSFT.csv'
meta_file = 'META.csv'
qqq_file = 'QQQ.csv'

# Read CSV files
aapl = pd.read_csv(aapl_file, usecols=['Date', 'Price'], parse_dates=['Date'], index_col='Date')
amzn = pd.read_csv(amzn_file, usecols=['Date', 'Price'], parse_dates=['Date'], index_col='Date')
googl = pd.read_csv(googl_file, usecols=['Date', 'Price'], parse_dates=['Date'], index_col='Date')
msft = pd.read_csv(msft_file, usecols=['Date', 'Price'], parse_dates=['Date'], index_col='Date')
meta = pd.read_csv(meta_file, usecols=['Date', 'Price'], parse_dates=['Date'], index_col='Date')
qqq = pd.read_csv(qqq_file, usecols=['Date', 'Price'], parse_dates=['Date'], index_col='Date')

# Rename columns
aapl.rename(columns={'Price': 'AAPL'}, inplace=True)
amzn.rename(columns={'Price': 'AMZN'}, inplace=True)
googl.rename(columns={'Price': 'GOOGL'}, inplace=True)
msft.rename(columns={'Price': 'MSFT'}, inplace=True)
meta.rename(columns={'Price': 'META'}, inplace=True)
qqq.rename(columns={'Price': 'QQQ'}, inplace=True)

# Merge data on Date
stocks = aapl.join([amzn, googl, msft, meta, qqq], how='inner')

# Compute correlation
correlation_matrix = stocks.corr()

# Plot correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Stock Price Correlation')
plt.show()

# Plot scatterplots for correlation
sns.pairplot(stocks)
plt.show()


# Plot stock prices from 2020 -2025
stocks.plot(figsize=(12, 6), title='Stock Price Trends')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend(title='Stocks')
plt.show()