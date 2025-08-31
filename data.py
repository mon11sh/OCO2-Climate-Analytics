# oco2_downloader.py
import earthaccess
from pathlib import Path
from typing import List

def login_to_earthdata():
    """
    Logs into NASA Earthdata using credentials stored in netrc or prompted interactively.
    """
    return earthaccess.login()

def fetch_oco2_data(
    output_dir: str = "./oco2_downloads",
    years: List[int] = list(range(2018, 2025)),
    short_name: str = "OCO2_L2_Lite_FP"
):
    """
    Download OCO-2 Lite data for a list of years if not already downloaded.
    
    Args:
        output_dir (str): Directory to save downloaded files.
        years (List[int]): List of years to fetch data for.
        short_name (str): Dataset short name (default: OCO2_L2_Lite_FP).
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    earthaccess.login()

    for year in years:
        day = f"{year}-07-01"
        print(f"🔎 Checking OCO-2 data for {day}")

        results = earthaccess.search_data(
            short_name=short_name,
            temporal=(day, day),
            bounding_box=(-180, -90, 180, 90),
        )

        if not results:
            print(f"⚠️ No data found for {day}")
            continue

        # Check if all files already exist
        all_exist = True
        for r in results:
            filename = Path(r.data_links()[0]).name  # first download link
            if not (output_path / filename).exists():
                all_exist = False
                break

        if all_exist:
            print(f"✅ Files for {day} already downloaded, skipping.")
            continue

        print(f"⬇️ Downloading new files for {day}")
        earthaccess.download(results, output_dir)

    print(f"🎉 Download complete. Files saved in: {output_dir}")
