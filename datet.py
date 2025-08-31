import pandas as pd

# Load the combined CSV
df = pd.read_csv("./oco2_ingested/combined_oco2_data.csv")

# Display first few rows with the date column
print(df[["source_file", "date"]].head())

# Check data types and info
print(df.info())

# Convert to datetime (if not already)
df["date"] = pd.to_datetime(df["date"])
print(df["date"].head())

# ✅ Get unique dates
print(df["date"].unique())

# Or if you want sorted unique values
print(df["date"].sort_values().unique())
