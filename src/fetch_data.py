import glob
import os
import sys
import time
from datetime import date, timedelta

import requests
import yaml

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY = (
    "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,"
    "relative_humidity_2m_mean,pressure_msl_mean,weathercode"
)
START_DATE = "2015-01-01"
DAYS_BEHIND = 7        # the newest days in the archive are not complete yet, so stop a week before today


def read_city_coordinates(raw_dir):
    """Latitude and longitude of every city, taken from line 2 of its existing CSV file."""
    coordinates = {}
    for file in sorted(glob.glob(os.path.join(raw_dir, "*_weather.csv"))):
        city = os.path.basename(file).split("_weather")[0]
        with open(file, encoding="utf-8") as f:
            f.readline()                                          # line 1: column names
            latitude, longitude = f.readline().split(",")[:2]     # line 2: the values
        coordinates[city] = (float(latitude), float(longitude))
    return coordinates


def download_city(city, latitude, longitude, end_date):
    """Download the whole history of one city as CSV text. Try up to 5 times before giving up."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": end_date,
        "daily": DAILY,
        "timezone": "GMT",
        "format": "csv",
    }
    for attempt in range(1, 6):
        try:
            response = requests.get(ARCHIVE_URL, params=params, timeout=60)
            if response.status_code == 429:               # the free API only allows so many calls per minute
                print(f"{city}: rate limit reached, waiting 65 seconds")
                time.sleep(65)
                continue
            response.raise_for_status()
            return response.text
        except requests.RequestException as error:
            print(f"{city}: attempt {attempt} failed ({error})")
            time.sleep(5)
    raise RuntimeError(f"Could not download the data for {city}")


def check_csv(text, city, end_date):
    """Stop if the downloaded text is not a complete CSV in the same layout as our files."""
    lines = text.strip().splitlines()
    if len(lines) < 5 or not lines[0].startswith("latitude,longitude") or not lines[3].startswith("time,"):
        raise ValueError(f"{city}: the response is not in the expected CSV layout")

    rows = lines[4:]                                              # lines 1-4 are the header lines
    expected_days = (date.fromisoformat(end_date) - date.fromisoformat(START_DATE)).days + 1
    if len(rows) != expected_days:
        raise ValueError(f"{city}: expected {expected_days} days but got {len(rows)}")
    if not rows[0].startswith(START_DATE) or not rows[-1].startswith(end_date):
        raise ValueError(f"{city}: the first or the last date is wrong")

    for row in rows:
        values = row.split(",")
        if len(values) != 8 or "" in values:
            raise ValueError(f"{city}: bad or empty value in row: {row}")


def main():
    # optional argument: a different folder to save into (useful for testing)
    out_dir = sys.argv[1] if len(sys.argv) > 1 else None

    with open("config/train.yaml") as f:
        cfg = yaml.safe_load(f)
    raw_dir = cfg["data"]["raw_dir"]
    out_dir = out_dir or raw_dir

    end_date = (date.today() - timedelta(days=DAYS_BEHIND)).isoformat()
    coordinates = read_city_coordinates(raw_dir)
    if not coordinates:
        raise FileNotFoundError(f"No *_weather.csv files found in {raw_dir} to read the city coordinates from")
    print(f"Downloading {START_DATE} to {end_date} for {len(coordinates)} cities")

    # download and check everything first, so a failure never leaves half-updated files
    downloaded = {}
    for city, (latitude, longitude) in coordinates.items():
        text = download_city(city, latitude, longitude, end_date)
        check_csv(text, city, end_date)
        downloaded[city] = text
        print(f"{city}: ok, {len(text.strip().splitlines()) - 4} days")
        time.sleep(5)                                             # be polite to the free API

    os.makedirs(out_dir, exist_ok=True)
    for city, text in downloaded.items():
        with open(os.path.join(out_dir, f"{city}_weather.csv"), "w", encoding="utf-8", newline="") as f:
            f.write(text)
    print(f"Saved {len(downloaded)} files to {out_dir}")


if __name__ == "__main__":
    main()
