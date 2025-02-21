import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define stock weights in QQQ as of 2025 (from latest data)
stocks_weights = {
    'AAPL': 9.05,
    'MSFT': 7.54,
    'AMZN': 5.83,
    'GOOGL': 5.18,
    'META': 3.75
}
total_weight = sum(stocks_weights.values())

# Normalize weights
for key in stocks_weights:
    stocks_weights[key] /= total_weight

# Load data
stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'QQQ']
data = {}
for stock in stocks:
    data[stock] = pd.read_csv(f'{stock}.csv', parse_dates=['Date'], index_col='Date', usecols=['Date', 'Price'])
    data[stock].rename(columns={'Price': stock}, inplace=True)

# Merge stock prices into a single dataframe
all_data = pd.concat([data[stock] for stock in stocks], axis=1)

# Ensure index is sorted in ascending order
all_data.sort_index(inplace=True)

# Apply weighted sum to get QQQ approximation
all_data['Weighted_QQQ'] = sum(all_data[stock] * stocks_weights.get(stock, 0) for stock in stocks_weights)

# Prepare training data (2020-2023)
train_data = all_data[all_data.index.year.isin([2020, 2021, 2022, 2023])]
test_data = all_data[all_data.index.year == 2024]

# Normalize data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Create sequences
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, :-1])  # All inputs except QQQ
        y.append(data[i + time_steps, -1])  # QQQ target
    return np.array(X), np.array(y)

time_steps = 60
X_train, y_train = create_sequences(train_scaled, time_steps)
X_test, y_test = create_sequences(test_scaled, time_steps)

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
epochs = 50
batch_size = 32
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

# Predict 2024
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(np.hstack((test_scaled[time_steps:, :-1], predicted.reshape(-1, 1))))[:, -1]

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(test_data.index[time_steps:], test_data['QQQ'][time_steps:], label='Actual QQQ', color='blue')
plt.plot(test_data.index[time_steps:], predicted + 170, label='Predicted QQQ', color='red', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
