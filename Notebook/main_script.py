# main_script.py

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and prepare data
def load_and_prepare_data():
    brent_oil_data = pd.read_csv('Data/BrentOilPrices.csv')
    gold_data = pd.read_csv('Data/gld_price_data.csv')
    silver_data = pd.read_csv('Data/LBMA-SILVER.csv')

    # Convert dates to datetime format
    brent_oil_data['Date'] = pd.to_datetime(brent_oil_data['Date'], errors='coerce')
    gold_data['Date'] = pd.to_datetime(gold_data['Date'], format='%m/%d/%Y')
    silver_data['Date'] = pd.to_datetime(silver_data['Date'])

    # Drop unnecessary columns
    gold_data = gold_data.drop(columns=['EUR/USD', 'USO', 'SPX'])
    silver_data = silver_data.drop(columns=['USD', 'GBP', 'EURO'])

    # Merge datasets
    merged_data = pd.merge(brent_oil_data, gold_data, on='Date', how='inner')
    merged_data = pd.merge(merged_data, silver_data, on='Date', how='inner')

    # Round prices to 3 decimal places
    merged_data['OIL'] = merged_data['OIL'].round(3)
    merged_data['GLD'] = merged_data['GLD'].round(3)
    merged_data['SLV'] = merged_data['SLV'].round(3)

    return merged_data

# Function to perform EDA
def exploratory_data_analysis(merged_data):
    # Plot Distribution of Prices
    plt.figure(figsize=(15, 5))
    sns.histplot(merged_data['OIL'], kde=True, label='Oil Price', color='brown')
    sns.histplot(merged_data['GLD'], kde=True, label='Gold Price', color='orange')
    sns.histplot(merged_data['SLV'], kde=True, label='Silver Price', color='grey')
    plt.legend()
    plt.title('Distribution of Commodity Prices')
    plt.xlabel('Price')
    plt.ylabel('Density')
    plt.show()

    # Plot Price Trends Over Time
    plt.figure(figsize=(10, 6))
    plt.plot(merged_data['Date'], merged_data['OIL'], label='Oil Price', color='brown')
    plt.plot(merged_data['Date'], merged_data['GLD'], label='Gold Price', color='orange')
    plt.plot(merged_data['Date'], merged_data['SLV'], label='Silver Price', color='grey')
    plt.title('Price Trends Over Time (2008-2018)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Correlation Analysis
    correlation_matrix = merged_data[['OIL', 'GLD', 'SLV']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=1, linecolor='black')
    plt.title('Correlation Matrix of Commodity Prices')
    plt.show()

# Function to calculate RMSE and MAE
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

# Function to run Prophet model
def run_prophet_model(df, column):
    df_prophet = df[['Date', column]].rename(columns={'Date': 'ds', column: 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)
    return forecast, model

# Function to run ARIMA model
def run_arima_model(series, order=(5, 1, 0)):
    model = ARIMA(series, order=order)
    result = model.fit()
    forecast = result.forecast(steps=365)
    return result, forecast

# Function to run LSTM model
def run_lstm_model(series, seq_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_series = scaler.fit_transform(series.values.reshape(-1, 1))

    # Create sequences for LSTM
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_series, seq_length)

    # Split data into training and testing sets
    train_size = int(X.shape[0] * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile and train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=10)

    # Make predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    return predictions, y_test, train_size

# Main function to execute the script
if __name__ == "__main__":
    # Load and prepare data
    merged_data = load_and_prepare_data()

    # Save the merged dataset as CSV
    merged_data.to_csv('Clean Data/merged_data.csv', index=False)

    # Perform EDA
    exploratory_data_analysis(merged_data)

    # Forecasting with Prophet
    forecast_oil, model_oil = run_prophet_model(merged_data, 'OIL')
    forecast_gold, model_gold = run_prophet_model(merged_data, 'GLD')
    forecast_silver, model_silver = run_prophet_model(merged_data, 'SLV')

    # Calculate Prophet metrics
    prophet_rmse_oil, prophet_mae_oil = calculate_metrics(merged_data['OIL'], forecast_oil['yhat'][:len(merged_data)])
    prophet_rmse_gold, prophet_mae_gold = calculate_metrics(merged_data['GLD'], forecast_gold['yhat'][:len(merged_data)])
    prophet_rmse_silver, prophet_mae_silver = calculate_metrics(merged_data['SLV'], forecast_silver['yhat'][:len(merged_data)])

    # Run ARIMA model for Silver
    arima_result, arima_forecast_silver = run_arima_model(merged_data['SLV'])
    arima_rmse_silver, arima_mae_silver = calculate_metrics(merged_data['SLV'], arima_result.predict(start=0, end=len(merged_data['SLV'])-1))

    # Run LSTM model for Silver
    predictions, y_test, train_size = run_lstm_model(merged_data['SLV'])
    lstm_rmse_silver, lstm_mae_silver = calculate_metrics(merged_data['SLV'][train_size + 60:], predictions)

    # Print metrics
    print(f"Prophet Model - Oil: RMSE = {prophet_rmse_oil}, MAE = {prophet_mae_oil}")
    print(f"Prophet Model - Gold: RMSE = {prophet_rmse_gold}, MAE = {prophet_mae_gold}")
    print(f"Prophet Model - Silver: RMSE = {prophet_rmse_silver}, MAE = {prophet_mae_silver}")
    print(f"ARIMA Model - Silver: RMSE = {arima_rmse_silver}, MAE = {arima_mae_silver}")
    print(f"LSTM Model - Silver: RMSE = {lstm_rmse_silver}, MAE = {lstm_mae_silver}")
