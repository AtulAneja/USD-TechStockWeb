import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load QQQ data
data = pd.read_csv('/content/drive/MyDrive/FINAL PROJECT STOCKS/QQQ.csv', parse_dates=['Date'], index_col='Date', usecols=['Date', 'Price'])
data.rename(columns={'Price': 'QQQ'}, inplace=True)

# Ensure index is sorted in ascending order
data.sort_index(inplace=True)

# Prepare training data (2020-2023)
train_data = data[data.index.year.isin([2020, 2021, 2022, 2023])]
test_data = data[data.index.year == 2024]

# Normalize data
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Create sequences
def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, 0])  # Use only QQQ column
        y.append(data[i + time_steps, 0])  # Predict QQQ itself
    return np.array(X).reshape(-1, time_steps, 1), np.array(y)

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
predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).flatten()

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(test_data.index[time_steps:], test_data['QQQ'][time_steps:], label='Actual QQQ', color='blue')
plt.plot(test_data.index[time_steps:], predicted, label='Predicted QQQ', color='red', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
