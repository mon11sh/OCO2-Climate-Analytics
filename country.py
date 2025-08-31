# country.py
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def aggregate_country_daily(input_file: str,
                            shapefile_path: str,
                            output_file: str = "./oco2_ingested/country_daily_co2.csv") -> pd.DataFrame:
    """
    Assign each CO₂ measurement to a country and compute daily country-level averages.

    Parameters
    ----------
    input_file : str
        Path to cleaned CO₂ data CSV (must include 'longitude', 'latitude', 'xco2', 'date').
    shapefile_path : str
        Path to Natural Earth shapefile with country polygons.
    output_file : str, optional
        Path to save country-level aggregated data (default: './oco2_ingested/country_daily_co2.csv').

    Returns
    -------
    pd.DataFrame
        DataFrame containing daily average xco2 by country.
    """

    # Load cleaned CO2 data
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])

    # Create GeoDataFrame from lat/lon
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    # Load world country polygons
    world = gpd.read_file(shapefile_path)

    # Spatial join: assign points to countries
    gdf_with_country = gpd.sjoin(
        gdf,
        world[['geometry', 'NAME']],
        how='left',
        predicate='within'
    )

    # Rename and aggregate
    gdf_with_country.rename(columns={'NAME': 'country'}, inplace=True)
    country_daily = (
        gdf_with_country
        .groupby(['country', 'date'])['xco2']
        .mean()
        .reset_index()
    )

    # Drop measurements not inside any country (e.g., ocean)
    country_daily = country_daily.dropna(subset=['country'])

    # Save
    country_daily.to_csv(output_file, index=False)
    print(f"✅ Country-level daily CO₂ aggregation saved to {output_file}")

    return country_daily


# Example usage
if __name__ == "__main__":
    shapefile = r"C:/code_1/earth_one/data/naturalearth/ne_110m_admin_0_countries.shp"
    result = aggregate_country_daily(
        input_file="./oco2_ingested/cleaned_oco2_data.csv",
        shapefile_path=shapefile
    )
    print(result.head())
    # Example: filter India
    print(result[result['country'] == "India"].head())
