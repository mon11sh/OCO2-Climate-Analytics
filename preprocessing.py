# oco2_preprocess.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def preprocess_oco2_data(
    input_csv: str = "./oco2_ingested/combined_oco2_data.csv",
    output_csv: str = "./oco2_ingested/cleaned_oco2_data.csv",
    plot: bool = True
) -> pd.DataFrame:
    """
    Preprocess OCO-2 combined dataset:
      - Convert dates
      - Drop missing values
      - Filter by valid ranges
      - Save cleaned dataset
      - Optionally plot spatial distribution
    
    Args:
        input_csv (str): Path to combined OCO-2 data CSV.
        output_csv (str): Path to save cleaned CSV.
        plot (bool): Whether to show a scatter plot of xco2 distribution.
    
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Load Data
    df = pd.read_csv(input_csv)
    df["date"] = pd.to_datetime(df["date"])

    print("✅ Loaded data")
    print(df.info())

    # Cleaning
    df_clean = df.dropna(subset=["xco2", "latitude", "longitude"])
    df_clean = df_clean[(df_clean["xco2"] >= 350) & (df_clean["xco2"] <= 500)]
    df_clean = df_clean[(df_clean["latitude"] >= -90) & (df_clean["latitude"] <= 90)]
    df_clean = df_clean[(df_clean["longitude"] >= -180) & (df_clean["longitude"] <= 180)]

    print(f"✅ Cleaned dataset rows: {len(df_clean)}")

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_csv, index=False)
    print(f"💾 Cleaned data saved to: {output_csv}")

    # Optional plotting
    if plot:
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            df_clean["longitude"], df_clean["latitude"],
            c=df_clean["xco2"], cmap="viridis", s=1
        )
        plt.colorbar(scatter, label="xco2 (ppm)")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Spatial distribution of CO₂ measurements")
        plt.show()

    return df_clean


# Example manual run
if __name__ == "__main__":
    preprocess_oco2_data()
