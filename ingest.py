# oco2_ingestor.py
import os
import xarray as xr
import pandas as pd
from pathlib import Path

def ingest_data(
    data_folder: str,
    output_folder: str = "./oco2_ingested",
    overwrite: bool = False
) -> str:
    """
    Ingest OCO-2 NetCDF/HDF files into a single cleaned CSV file.

    Args:
        data_folder (str): Path to the folder containing downloaded .nc4/.h5 files.
        output_folder (str): Path to the folder to save processed CSV.
        overwrite (bool): If True, reprocess data even if output already exists.

    Returns:
        str: Path to the combined CSV file.
    """
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_folder) / "combined_oco2_data.csv"

    if output_path.exists() and not overwrite:
        print(f"✅ Ingestion skipped. File already exists: {output_path}")
        return str(output_path)

    data_files = [f for f in os.listdir(data_folder) if f.endswith((".nc4", ".h5"))]
    if not data_files:
        print("⚠️ No data files found to ingest.")
        return ""

    all_dfs = []
    for file in data_files:
        filepath = os.path.join(data_folder, file)
        print(f"📂 Processing file: {filepath}")
        try:
            ds = xr.open_dataset(filepath, decode_times=True)
            df = pd.DataFrame({
                "xco2": ds["xco2"].values,
                "latitude": ds["latitude"].values,
                "longitude": ds["longitude"].values,
                "time": pd.to_datetime(ds["time"].values),
            })
            df["date"] = df["time"].dt.date
            df_clean = df.dropna(subset=["xco2", "latitude", "longitude", "date"])
            df_clean["source_file"] = file
            all_dfs.append(df_clean)
        except Exception as e:
            print(f"❌ Failed to process {file}: {e}")
            continue

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(output_path, index=False)
        print(f"🎉 Ingestion complete. Combined data saved to {output_path}")
        return str(output_path)
    else:
        print("⚠️ No valid data ingested.")
        return ""
