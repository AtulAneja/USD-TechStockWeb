import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import ta
import os

def load_data(symbol):
    """Load and preprocess stock data"""
    df = pd.read_csv(f'{symbol}.csv')
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Convert Volume (Vol.) to numeric, removing 'M' and multiplying by 1,000,000
    df['Volume'] = df['Vol.'].str.replace('M', '').astype(float) * 1_000_000
    
    # Convert Change % to numeric, removing '%' and dividing by 100
    df['Change'] = df['Change %'].str.replace('%', '').astype(float) / 100
    
    # Drop original Vol. and Change % columns
    df = df.drop(['Vol.', 'Change %'], axis=1)
    
    # Sort by date
    df = df.sort_values('Date')
    
    print(f"{symbol} data loaded: {len(df)} rows from {df['Date'].min()} to {df['Date'].max()}")
    return df

def plot_all_stocks_with_realistic_prediction(stock_dfs, qqq_df):
    """Plot all stocks with realistic QQQ prediction for 2025 including corrections"""
    # Find common date range
    start_dates = [df['Date'].min() for df in stock_dfs.values()]
    start_dates.append(qqq_df['Date'].min())
    common_start_date = max(start_dates)
    
    end_dates = [df['Date'].max() for df in stock_dfs.values()]
    end_dates.append(qqq_df['Date'].max())
    common_end_date = min(end_dates)
    
    print(f"Common date range: {common_start_date} to {common_end_date}")
    
    # Filter data to common date range
    filtered_stock_dfs = {}
    for symbol, df in stock_dfs.items():
        filtered_stock_dfs[symbol] = df[(df['Date'] >= common_start_date) & (df['Date'] <= common_end_date)].copy()
    
    filtered_qqq_df = qqq_df[(qqq_df['Date'] >= common_start_date) & (qqq_df['Date'] <= common_end_date)].copy()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: Actual prices
    for symbol, df in filtered_stock_dfs.items():
        ax1.plot(df['Date'], df['Price'], label=symbol, linewidth=1.5)
    
    ax1.plot(filtered_qqq_df['Date'], filtered_qqq_df['Price'], label='QQQ', linewidth=2.5, color='black')
    
    ax1.set_title('Stock Prices (Actual Values)', fontsize=16)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Normalized prices (starting at 100)
    for symbol, df in filtered_stock_dfs.items():
        first_price = df['Price'].iloc[0]
        normalized_prices = df['Price'] / first_price * 100
        ax2.plot(df['Date'], normalized_prices, label=symbol, linewidth=1.5)
    
    qqq_first_price = filtered_qqq_df['Price'].iloc[0]
    qqq_normalized = filtered_qqq_df['Price'] / qqq_first_price * 100
    ax2.plot(filtered_qqq_df['Date'], qqq_normalized, label='QQQ', linewidth=2.5, color='black')
    
    ax2.set_title('Normalized Stock Prices (Starting at 100)', fontsize=16)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Normalized Price', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Generate more realistic QQQ 2025 prediction with corrections
    last_price = filtered_qqq_df['Price'].iloc[-1]
    last_date = filtered_qqq_df['Date'].iloc[-1]
    
    # Calculate QQQ's statistics over the available period
    qqq_returns = filtered_qqq_df['Price'].pct_change().dropna()
    avg_daily_return = qqq_returns.mean()
    volatility = qqq_returns.std()
    
    # Generate future dates for 2025 (252 trading days)
    future_dates = []
    current_date = last_date
    trading_days = 252
    
    while len(future_dates) < trading_days:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # Weekdays only
            future_dates.append(current_date)
    
    # Generate more realistic future prices with market corrections
    np.random.seed(42)  # For reproducibility
    
    # Number of simulations to run
    num_simulations = 1000
    all_simulations = np.zeros((num_simulations, trading_days))
    
    for sim in range(num_simulations):
        current_price = last_price
        prices = []
        
        # Parameters for market corrections
        correction_probability = 0.005  # Probability of starting a correction on any given day
        in_correction = False
        correction_length = 0
        max_correction_length = 40
        correction_severity = np.random.uniform(0.1, 0.25)  # 10-25% correction
        
        for i in range(trading_days):
            if not in_correction:
                # Check if a correction starts
                if np.random.random() < correction_probability:
                    in_correction = True
                    correction_length = np.random.randint(15, max_correction_length)
                    correction_severity = np.random.uniform(0.1, 0.25)
                    # Calculate daily return during correction to reach target severity
                    correction_daily_return = (1 - correction_severity) ** (1 / correction_length) - 1
                
                # Normal trading day (not in correction)
                daily_return = np.random.normal(avg_daily_return, volatility)
                current_price = current_price * (1 + daily_return)
            else:
                # In correction period
                current_price = current_price * (1 + correction_daily_return)
                correction_length -= 1
                if correction_length <= 0:
                    in_correction = False
            
            prices.append(current_price)
        
        all_simulations[sim] = prices
    
    # Calculate median, 25th and 75th percentiles from all simulations
    median_simulation = np.median(all_simulations, axis=0)
    lower_bound = np.percentile(all_simulations, 25, axis=0)
    upper_bound = np.percentile(all_simulations, 75, axis=0)
    
    # Plot median prediction and confidence interval
    ax1.plot(future_dates, median_simulation, label='QQQ 2025 Prediction', color='red', linestyle='--', linewidth=2)
    ax1.fill_between(future_dates, lower_bound, upper_bound, color='red', alpha=0.2, label='50% Confidence Interval')
    
    # Add to normalized plot
    norm_median = median_simulation / qqq_first_price * 100
    norm_lower = lower_bound / qqq_first_price * 100
    norm_upper = upper_bound / qqq_first_price * 100
    
    ax2.plot(future_dates, norm_median, label='QQQ 2025 Prediction', color='red', linestyle='--', linewidth=2)
    ax2.fill_between(future_dates, norm_lower, norm_upper, color='red', alpha=0.2)
    
    # Add vertical line at prediction start
    ax1.axvline(x=last_date, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=last_date, color='gray', linestyle='--', alpha=0.5)
    
    # Add text annotation
    ax1.text(last_date, ax1.get_ylim()[0], 'Prediction Start', rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig('all_stocks_with_realistic_prediction.png', dpi=300, bbox_inches='tight')
    
    # Print key prediction metrics
    print(f"\nLast actual QQQ price: ${last_price:.2f}")
    print(f"Predicted QQQ price at end of 2025: ${median_simulation[-1]:.2f}")
    print(f"Predicted QQQ price range: ${lower_bound[-1]:.2f} - ${upper_bound[-1]:.2f}")
    print(f"Predicted QQQ average price for 2025: ${np.mean(median_simulation):.2f}")
    print(f"Predicted growth for 2025: {((median_simulation[-1] / last_price) - 1) * 100:.2f}%")
    
    # Create separate plot showing multiple possible paths
    plt.figure(figsize=(20, 10))
    
    # Plot historical data
    plt.plot(filtered_qqq_df['Date'], filtered_qqq_df['Price'], 
             label='QQQ Historical', color='blue', linewidth=2)
    
    # Plot median prediction
    plt.plot(future_dates, median_simulation, 
             label='Median Prediction', color='red', linewidth=2, linestyle='--')
    
    # Plot confidence interval
    plt.fill_between(future_dates, lower_bound, upper_bound, 
                    color='red', alpha=0.2, label='50% Confidence Interval')
    
    # Plot a few sample paths
    num_samples = 5
    for i in range(num_samples):
        sample_idx = np.random.randint(0, num_simulations)
        plt.plot(future_dates, all_simulations[sample_idx], 
                 alpha=0.5, linewidth=0.8, color='gray')
    
    plt.title('QQQ Prediction for 2025 with Sample Paths', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add vertical line at prediction start
    plt.axvline(x=last_date, color='gray', linestyle='--', alpha=0.5)
    plt.text(last_date, plt.ylim()[0], 'Prediction Start', rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig('qqq_multiple_scenarios_2025.png', dpi=300, bbox_inches='tight')
    
    # Save prediction data
    prediction_df = pd.DataFrame({
        'Date': future_dates,
        'Median_Prediction': median_simulation,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound
    })
    prediction_df.to_csv('qqq_predictions_2025.csv', index=False)
    
    return future_dates, median_simulation

def main():
    # Load all data
    print("Loading stock data...")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN']
    stock_dfs = {}
    
    for symbol in symbols:
        stock_dfs[symbol] = load_data(symbol)
    
    # Load QQQ data
    print("\nLoading QQQ data...")
    qqq_df = load_data('QQQ')
    
    # Plot all stocks with realistic QQQ prediction
    print("\nCreating visualization with all stocks and realistic QQQ prediction...")
    future_dates, median_prediction = plot_all_stocks_with_realistic_prediction(stock_dfs, qqq_df)
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()
