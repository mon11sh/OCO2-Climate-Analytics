# pipeline.py
import os
from data import fetch_oco2_data
from ingest import ingest_data
from preprocessing import preprocess_oco2_data
from aggregation import aggregate_global_lat_bands
from country import aggregate_country_daily
from arima import arima_forecast_country
from lstm import lstm_forecast_country

# === CONFIG ===
data_folder = "./oco2_downloads"
ingested_folder = "./oco2_ingested"
combined_csv = os.path.join(ingested_folder, "combined_oco2_data.csv")
country_csv = os.path.join(ingested_folder, "country_daily_co2.csv")
country_name = "India"  # Change as needed

def run_pipeline():
    # STEP 0: Fetch raw OCO-2 data
    print("\n=== STEP 0: FETCH RAW DATA ===")
    os.makedirs(data_folder, exist_ok=True)
    fetch_oco2_data(output_dir=data_folder)

    
    # STEP 1: Ingest
    print("\n=== STEP 1: INGEST DATA ===")
    ingest_data(data_folder, ingested_folder)

    # STEP 2: Preprocess
    print("\n=== STEP 2: PREPROCESS DATA ===")
    preprocess_data(ingested_folder)

    # STEP 3: Aggregate
    print("\n=== STEP 3: AGGREGATE DATA ===")
    aggregate_data(ingested_folder, combined_csv)

    # STEP 4: Country-level aggregation
    print("\n=== STEP 4: COUNTRY AGGREGATION ===")
    aggregate_country_daily(combined_csv, country_csv)

    # STEP 5: ARIMA Forecast
    print("\n=== STEP 5: ARIMA FORECAST ===")
    arima_forecast_country(country_csv, country_name, forecast_months=12,
                           save_plot=f"{country_name.lower()}_arima_forecast.png")

    # STEP 6: LSTM Forecast
    print("\n=== STEP 6: LSTM FORECAST ===")
    lstm_forecast_country(country_csv, country_name, n_steps=4, forecast_horizon=4,
                          epochs=50, output_dir=ingested_folder)

    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
