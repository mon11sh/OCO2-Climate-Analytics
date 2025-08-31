# arima.py
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

def arima_forecast_country(
    csv_path: str,
    country: str,
    forecast_months: int = 12,
    save_plot: str | None = None
):
    """
    Fit an ARIMA (SARIMA) model to forecast CO₂ for a given country.

    Parameters
    ----------
    csv_path : str
        Path to country-level daily CO₂ CSV (must include 'country', 'date', 'xco2').
    country : str
        Name of the country to forecast.
    forecast_months : int, default=12
        Number of months to forecast into the future.
    save_plot : str | None, default=None
        If a file path is provided, saves the forecast plot as an image.

    Returns
    -------
    pd.Series
        Forecasted monthly mean CO₂ values with datetime index.
    """

    # Load country aggregated CO₂ data
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    # Filter for the specified country
    country_df = df[df['country'] == country].copy()
    if country_df.empty:
        raise ValueError(f"No data found for country: {country}")

    # Set date as index and sort
    country_df.set_index('date', inplace=True)
    country_df.sort_index(inplace=True)

    # Resample to monthly frequency with mean CO₂
    monthly_co2 = country_df['xco2'].resample('M').mean()

    # Data sufficiency check
    if len(monthly_co2) < 24:
        print("⚠️ Warning: Less than 2 years of data may limit forecast accuracy.")

    # Fit SARIMA model (seasonal order 12 = yearly seasonality)
    model = SARIMAX(
        monthly_co2,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)

    # Forecast next n months
    forecast = results.get_forecast(steps=forecast_months)
    forecast_ci = forecast.conf_int()

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_co2.index, monthly_co2, label='Observed')
    plt.plot(forecast.predicted_mean.index, forecast.predicted_mean,
             label='Forecast', color='red')
    plt.fill_between(
        forecast_ci.index,
        forecast_ci.iloc[:, 0],
        forecast_ci.iloc[:, 1],
        color='pink',
        alpha=0.3
    )
    plt.title(f'ARIMA Forecast of Monthly CO₂ for {country}')
    plt.xlabel('Date')
    plt.ylabel('CO₂ (ppm)')
    plt.legend()
    plt.grid(True)

    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches="tight")
        print(f"✅ Plot saved to {save_plot}")
    else:
        plt.show()

    return forecast.predicted_mean


# Example usage
if __name__ == "__main__":
    csv_path = './oco2_ingested/country_daily_co2.csv'
    country_name = 'India'
    forecast_values = arima_forecast_country(csv_path, country_name)
    print(forecast_values.head())
