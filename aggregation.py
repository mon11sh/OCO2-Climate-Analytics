# aggregation.py
import pandas as pd

def aggregate_global_lat_bands(
    input_file: str,
    output_global: str = "./oco2_ingested/daily_global_mean.csv",
    output_latband: str = "./oco2_ingested/daily_latband_mean.csv"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute daily global mean CO₂ and daily mean CO₂ by latitude band.

    Parameters
    ----------
    input_file : str
        Path to cleaned CO₂ data CSV (must include 'date', 'latitude', 'xco2').
    output_global : str, optional
        File path to save global daily mean (default: './oco2_ingested/daily_global_mean.csv').
    output_latband : str, optional
        File path to save latitude-band daily mean (default: './oco2_ingested/daily_latband_mean.csv').

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        - global_daily: daily mean CO₂ (global)
        - latband_daily: daily mean CO₂ by latitude band
    """

    # Load the cleaned, ingested CSV
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])

    # 1. Daily global mean CO₂
    global_daily = df.groupby('date')['xco2'].mean().reset_index()

    # 2. Latitude bands (every 30°)
    lat_bins = [-90, -60, -30, 0, 30, 60, 90]
    lat_labels = ['-90 to -60', '-60 to -30', '-30 to 0',
                  '0 to 30', '30 to 60', '60 to 90']

    df['lat_band'] = pd.cut(df['latitude'],
                            bins=lat_bins,
                            labels=lat_labels,
                            include_lowest=True)

    latband_daily = (
        df.groupby(['lat_band', 'date'], observed=True)['xco2']
        .mean()
        .reset_index()
    )

    # Save outputs
    global_daily.to_csv(output_global, index=False)
    latband_daily.to_csv(output_latband, index=False)

    print(f"✅ Saved global daily mean to {output_global}")
    print(f"✅ Saved latitude-band daily mean to {output_latband}")

    return global_daily, latband_daily


# Example usage
if __name__ == "__main__":
    g, l = aggregate_global_lat_bands(
        input_file="./oco2_ingested/cleaned_oco2_data.csv"
    )
    print("Global daily mean sample:")
    print(g.head())
    print("\nLatitude-band daily mean sample:")
    print(l.head())
