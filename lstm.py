# lstm.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os


def prepare_lstm_data(series, n_steps):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i+n_steps])
        y.append(series[i+n_steps])
    return np.array(X), np.array(y)


def lstm_forecast_country(csv_path, country, n_steps=4, forecast_horizon=4, epochs=50, output_dir="./oco2_ingested"):
    # Load and preprocess data
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    # Validate country
    if country not in df['country'].unique():
        raise ValueError(f"Country '{country}' not found in dataset. Available countries: {df['country'].unique()}")

    country_df = df[df['country'] == country].copy()
    country_df.set_index('date', inplace=True)
    country_df.sort_index(inplace=True)

    # Use raw data points without resampling due to sparse irregular dates
    monthly_co2 = country_df['xco2'].copy()
    monthly_co2.index = country_df.index

    print(f"Number of data points: {len(monthly_co2)}")

    if len(monthly_co2) < 2:
        raise ValueError("Not enough data points for LSTM modeling.")

    # Scale data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(monthly_co2.values.reshape(-1, 1))

    # Adjust n_steps
    adjusted_n_steps = min(n_steps, len(data_scaled) - 1)
    if adjusted_n_steps < n_steps:
        print(f"⚠️ Reducing n_steps from {n_steps} to {adjusted_n_steps} due to limited data.")

    # Prepare sequences
    X, y = prepare_lstm_data(data_scaled, adjusted_n_steps)
    if X.size == 0:
        raise ValueError("Not enough sequences for LSTM. Reduce n_steps or add more data.")

    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(adjusted_n_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train
    history = model.fit(X, y, epochs=epochs, verbose=1)

    if np.isnan(history.history['loss']).any():
        raise ValueError("Training loss became NaN. Check input data.")

    # Forecast
    input_seq = data_scaled[-adjusted_n_steps:].reshape(adjusted_n_steps, 1)
    predictions = []
    for _ in range(forecast_horizon):
        pred = model.predict(input_seq.reshape(1, adjusted_n_steps, 1), verbose=0)
        predictions.append(pred[0, 0])
        input_seq = np.vstack([input_seq[1:], [[pred[0, 0]]]])

    # Inverse scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_co2.index, monthly_co2.values, label='Observed')
    forecast_index = pd.date_range(monthly_co2.index[-1], periods=forecast_horizon+1, freq='180D')[1:]
    plt.plot(forecast_index, predictions, label='LSTM Forecast', color='red')
    plt.title(f'LSTM Forecast of CO₂ for {country}')
    plt.xlabel('Date')
    plt.ylabel('CO₂ (ppm)')
    plt.legend()
    plt.grid(True)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{country.lower()}_lstm_forecast.png")
    plt.savefig(save_path)
    plt.close()

    print(f"✅ LSTM forecast saved to {save_path}")
    return predictions, save_path
