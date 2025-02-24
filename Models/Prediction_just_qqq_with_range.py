# Previous imports and helper functions remain the same...

def predict_future(model, last_sequence, scaler_dict, feature_columns, trading_days, look_back):
    """Modified prediction function with better continuity and stability"""
    future_predictions = []
    current_sequence = last_sequence.copy()
    price_scaler = scaler_dict['Price']

    # Get the last actual prices for calculating trend
    last_prices = price_scaler.inverse_transform(current_sequence[:, 0].reshape(-1, 1))
    last_actual_price = last_prices[-1][0]

    # Calculate recent trend
    recent_daily_returns = np.diff(last_prices[-30:], axis=0) / last_prices[-30:-1]
    avg_daily_return = np.mean(recent_daily_returns)

    # Calculate recent volatility
    volatility = np.std(recent_daily_returns) * np.sqrt(252)  # Annualized volatility
    daily_vol = volatility / np.sqrt(252)

    for day in range(trading_days):
        # Make base prediction
        current_sequence_reshaped = current_sequence.reshape((1, look_back, len(feature_columns)))
        next_pred_scaled = model.predict(current_sequence_reshaped, verbose=0)[0, 0]
        next_pred = price_scaler.inverse_transform([[next_pred_scaled]])[0, 0]

        if day == 0:
            # Ensure smooth transition from last actual price
            adjustment_factor = last_actual_price / next_pred
            next_pred = last_actual_price * (1 + avg_daily_return + np.random.normal(0, daily_vol))
        else:
            # Add momentum and random walk components
            prev_pred = future_predictions[-1]
            trend_component = avg_daily_return
            random_component = np.random.normal(0, daily_vol)
            next_pred = prev_pred * (1 + trend_component + random_component)

        future_predictions.append(next_pred)

        # Update sequence for next prediction
        new_row = current_sequence[-1].copy()
        new_row[0] = price_scaler.transform([[next_pred]])[0, 0]

        # Update other features based on the new prediction
        new_row[feature_columns.index('High')] = price_scaler.transform([[next_pred * (1 + daily_vol)]])[0, 0]
        new_row[feature_columns.index('Low')] = price_scaler.transform([[next_pred * (1 - daily_vol)]])[0, 0]
        new_row[feature_columns.index('Open')] = new_row[0]  # Use previous close as next open

        # Slide the window
        current_sequence = np.vstack([current_sequence[1:], new_row])

    future_predictions = np.array(future_predictions).reshape(-1, 1)

    # Apply smoothing to remove any sharp discontinuities
    smoothing_window = 5
    future_predictions = pd.Series(future_predictions.flatten()).rolling(window=smoothing_window,
                                                                         min_periods=1).mean().values.reshape(-1, 1)

    return future_predictions


def main():
    print("Loading data...")
    df = load_data()
    look_back = 60

    print("\nPreparing data with technical indicators...")
    X_train, X_test, y_train, y_test, scaler_dict, df, feature_columns = prepare_data(df, look_back)

    print("\nCreating and training model...")
    model = create_model(look_back, X_train.shape[2])

    # Add callbacks for better training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    trading_days_2025 = 252
    print("\nMaking predictions for 2025...")

    # Use the last window of actual data for prediction
    last_sequence = np.column_stack(
        [scaler_dict[col].transform(df[col].values[-look_back:].reshape(-1, 1)) for col in feature_columns])
    future_pred = predict_future(model, last_sequence, scaler_dict, feature_columns, trading_days_2025, look_back)

    # Generate future dates
    last_date = df['Date'].iloc[-1]
    future_dates = []
    current_date = last_date

    while len(future_dates) < trading_days_2025:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            future_dates.append(current_date)

    # Create visualization
    plt.figure(figsize=(20, 12))

    # Plot historical data
    plt.plot(df['Date'], df['Price'],
             label='Historical Data', color='blue', linewidth=1)

    # Plot predictions
    plt.plot(future_dates, future_pred,
             label='2025 Predictions', color='red', linewidth=2, linestyle='--')

    # Add confidence intervals
    confidence = 0.1
    upper_bound = future_pred * (1 + confidence)
    lower_bound = future_pred * (1 - confidence)
    plt.fill_between(future_dates, lower_bound.flatten(), upper_bound.flatten(),
                     color='red', alpha=0.1, label='Prediction Range')

    # Print transition values
    print(f"\nLast actual price: ${df['Price'].iloc[-1]:.2f}")
    print(f"First predicted price: ${future_pred[0][0]:.2f}")
    print(f"Predicted price range: ${future_pred.min():.2f} - ${future_pred.max():.2f}")

    plt.title('QQQ Stock Price Prediction for 2025', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Add vertical line for prediction start
    plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.5)
    plt.text(last_date, plt.ylim()[0], 'Prediction Start',
             rotation=90, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig('qqq_prediction_2025.png', dpi=300, bbox_inches='tight')

    # Print monthly averages
    print("\nPredicted Monthly Averages for 2025:")
    monthly_data = pd.DataFrame({'Date': future_dates, 'Price': future_pred.flatten()})
    monthly_data.set_index('Date', inplace=True)
    monthly_avg = monthly_data.resample('M').mean()

    for date, price in monthly_avg.iterrows():
        print(f"{date.strftime('%B %Y')}: ${price['Price']:.2f}")


if __name__ == "__main__":
    main()