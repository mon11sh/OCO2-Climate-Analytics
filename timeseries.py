# oco2_timeseries.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_timeseries(
    input_csv: str = "./oco2_ingested/cleaned_oco2_data.csv",
    plot: bool = True
) -> pd.DataFrame:
    """
    Generate a global daily mean CO₂ time series from OCO-2 data.

    Args:
        input_csv (str): Path to the cleaned OCO-2 dataset.
        plot (bool): Whether to generate plots.

    Returns:
        pd.DataFrame: DataFrame with ['date', 'xco2'] daily global means.
    """
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Load cleaned data
    df = pd.read_csv(input_csv)
    df["date"] = pd.to_datetime(df["date"])

    # Aggregate daily mean
    daily_mean = df.groupby("date")["xco2"].mean().reset_index()

    print(f"✅ Daily mean CO₂ computed for {len(daily_mean)} days")
    print("Date range:", daily_mean["date"].min(), "to", daily_mean["date"].max())

    # Optional plots
    if plot:
        # Line plot
        plt.figure(figsize=(10, 5))
        plt.plot(daily_mean["date"], daily_mean["xco2"], label="Global Mean xco2")
        plt.xlabel("Date")
        plt.ylabel("CO₂ (ppm)")
        plt.title("Global Daily Mean CO₂")
        plt.legend()
        plt.show()

        # Scatter plot
        plt.figure(figsize=(10, 5))
        plt.scatter(daily_mean["date"], daily_mean["xco2"], label="Global Mean xco2", s=10)
        plt.xlabel("Date")
        plt.ylabel("CO₂ (ppm)")
        plt.title("Global Daily Mean CO₂ (Scatter)")
        plt.legend()
        plt.show()

    return daily_mean


# Example manual run
if __name__ == "__main__":
    daily_mean = analyze_timeseries()
    print(daily_mean.head())
