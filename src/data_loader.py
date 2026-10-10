import glob
import os

import pandas as pd


def load_raw(cfg):
    """Read the city CSV files and join them into one table."""
    raw_dir = cfg["data"]["raw_dir"]

    if cfg["data"]["cities"] == "all":
        files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    else:
        files = [os.path.join(raw_dir, f"{city}_weather.csv") for city in cfg["data"]["cities"]]

    all_dfs = []
    for file in files:
        city_name = os.path.basename(file).split("_weather")[0]
        df_city = pd.read_csv(file, skiprows=cfg["data"]["skip_rows"])
        df_city.columns = cfg["data"]["columns"]
        df_city["city"] = city_name
        all_dfs.append(df_city)

    df = pd.concat(all_dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["city", "date"]).reset_index(drop=True)
    return df


def check_data(df):
    """Stop if there are duplicate days, missing values or gaps in the dates."""
    duplicates = df.duplicated(["city", "date"]).sum()
    nulls = df.isnull().sum().sum()
    gaps = df.groupby("city")["date"].diff().dt.days.dropna()
    gap_count = (gaps != 1).sum()

    print(f"Data check: duplicates={duplicates}, missing values={nulls}, date gaps={gap_count}")
    if duplicates > 0 or nulls > 0 or gap_count > 0:
        raise ValueError("Data check failed, see the numbers above")


def split_by_city(df, cfg):
    """Split each city in date order: first 70% train, next 15% validation, last 15% test."""
    train_share = cfg["split"]["train"]
    val_share = cfg["split"]["val"]

    train_list = []
    val_list = []
    test_list = []

    for city in df["city"].unique():
        city_df = df[df["city"] == city]       # this city only (already sorted by date)
        n = len(city_df)

        train_end = int(n * train_share)
        val_end = int(n * (train_share + val_share))

        train_list.append(city_df[:train_end])
        val_list.append(city_df[train_end:val_end])
        test_list.append(city_df[val_end:])

    train_df = pd.concat(train_list)
    val_df = pd.concat(val_list)
    test_df = pd.concat(test_list)
    return train_df, val_df, test_df
