import os
import sys
import time
from datetime import date, timedelta

import requests

START_DATE = "2015-01-01"
END_DATE = (date.today() - timedelta(days=7)).isoformat()   # archive is incomplete for the last week
DAILY = ("temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,"
         "relative_humidity_2m_mean,pressure_msl_mean,weathercode")

COORDINATES = {
    "Chennai": (13.110721, 80.2459),
    "Delhi": (28.576448, 77.18678),
    "Guwahati": (26.115992, 91.77437),
    "Jaipur": (26.889278, 75.83149),
    "Kolkata": (22.601053, 88.31776),
    "Mumbai": (19.086115, 72.85291),
    "Pune": (18.523726, 73.86876),
    "Shimla": (31.107204, 77.17339),
}


def download_city(city, latitude, longitude):
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={latitude}&longitude={longitude}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        f"&daily={DAILY}&timezone=GMT&format=csv"
    )
    response = requests.get(url, timeout=60)
    if response.status_code == 429:          # free API limit reached
        print(f"{city}: rate limit, waiting 65 seconds")
        time.sleep(65)
        response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.text


def main():
    out_dir = "data/raw"
    expected_days = (date.fromisoformat(END_DATE) - date.fromisoformat(START_DATE)).days + 1

    texts = {}
    for city, (latitude, longitude) in COORDINATES.items():
        text = download_city(city, latitude, longitude)
        
        
        days = len(text.strip().splitlines()) - 4      # first 4 lines are the header
        if days != expected_days:
            raise ValueError(f"{city}: got {days} days, expected {expected_days}")

        texts[city] = text
        print(f"{city}: done, {days} days")

    os.makedirs(out_dir, exist_ok=True)
    for city, text in texts.items():
        with open(os.path.join(out_dir, f"{city}_weather.csv"), "w", encoding="utf-8", newline="") as f:
            f.write(text)
    print(f"Saved {len(texts)} files to {out_dir}")


if __name__ == "__main__":
    main()
