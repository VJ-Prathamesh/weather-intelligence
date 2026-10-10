import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"        # hide TensorFlow's startup messages

import json
import sys
from datetime import date

import joblib
import numpy as np
import yaml
from tensorflow import keras

from src.data_loader import load_raw, split_by_city
from src.feature_engineering import make_windows

MIN_IMPROVEMENT = 0.05      # the LSTM must be at least 5% better than the baseline on average
MIN_TEMP = -20              # predictions outside this range (in C) are not believable
MAX_TEMP = 55
MAX_DATA_AGE_DAYS = 14      # the newest data must not be older than this

FILES = ["lstm_weather_model.keras", "scaler.pkl", "config.json", "metrics.json"]


def finish(results):
    """Print every check. Stop with an error if any check failed, so the pipeline stops too."""
    print("\nQuality gate")
    for name, passed, detail in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    if all(passed for _, passed, _ in results):
        print("\nGate passed: the model can be uploaded.")
        return
    print("\nGate FAILED: nothing will be uploaded.")
    sys.exit(1)


def check_beats_baseline_every_day(metrics):
    baseline = metrics["baseline_rmse"]
    lstm = metrics["lstm_rmse"]
    worse_days = [day + 1 for day in range(len(lstm)) if lstm[day] >= baseline[day]]
    if worse_days:
        return False, f"not better than the baseline on day(s) {worse_days}"
    return True, f"better than the baseline on all {len(lstm)} days"


def check_average_improvement(metrics):
    improvement = 1 - metrics["lstm_avg"] / metrics["baseline_avg"]
    detail = (f"LSTM {metrics['lstm_avg']} vs baseline {metrics['baseline_avg']}: "
              f"{improvement * 100:.1f}% better (need {MIN_IMPROVEMENT * 100:.0f}%)")
    return improvement >= MIN_IMPROVEMENT, detail


def check_shapes(model, scaler, config):
    window = config["window"]
    n_features = len(config["features"])
    horizon = config["horizon"]

    problems = []
    if model.input_shape != (None, window, n_features):
        problems.append(f"model input is {model.input_shape}, expected {(None, window, n_features)}")
    if model.output_shape != (None, horizon):
        problems.append(f"model output is {model.output_shape}, expected {(None, horizon)}")
    if scaler.n_features_in_ != n_features:
        problems.append(f"scaler has {scaler.n_features_in_} features, expected {n_features}")

    if problems:
        return False, "; ".join(problems)
    return True, f"input (30 days x {n_features} features) and output ({horizon} days) look right"


def check_predictions(model, scaler, config, df, cfg):
    """Predict on the test part of the data and check the numbers are believable."""
    features = config["features"]
    _, _, test_df = split_by_city(df, cfg)
    test_df = test_df.copy()
    test_df[features] = scaler.transform(test_df[features])
    X_test, _ = make_windows(test_df, features, config["window"], config["horizon"])

    scaled = model.predict(X_test, verbose=0)
    predictions = scaled * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]   # back to C

    if not np.isfinite(predictions).all():
        return False, "some predictions are NaN or infinite"
    low, high = predictions.min(), predictions.max()
    if low < MIN_TEMP or high > MAX_TEMP:
        return False, f"predictions go from {low:.1f} to {high:.1f} C, outside {MIN_TEMP} to {MAX_TEMP}"
    return True, f"{len(predictions)} predictions, from {low:.1f} to {high:.1f} C"


def check_fresh_data(df):
    newest = df["date"].max().date()
    age = (date.today() - newest).days
    return age <= MAX_DATA_AGE_DAYS, f"newest data is {newest} ({age} days old, limit {MAX_DATA_AGE_DAYS})"


def main():
    with open("config/train.yaml") as f:
        cfg = yaml.safe_load(f)

    # optional arguments: the folder with the trained files, and the folder with the CSV data
    model_dir = sys.argv[1] if len(sys.argv) > 1 else cfg["output"]["dir"]
    if len(sys.argv) > 2:
        cfg["data"]["raw_dir"] = sys.argv[2]

    results = []

    # 1. the four files must exist
    missing = [name for name in FILES if not os.path.exists(os.path.join(model_dir, name))]
    results.append(("The 4 files exist", not missing, f"missing: {', '.join(missing)}" if missing else model_dir))
    if missing:
        finish(results)

    # 2. the files must load
    try:
        model = keras.models.load_model(os.path.join(model_dir, "lstm_weather_model.keras"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
        with open(os.path.join(model_dir, "config.json")) as f:
            config = json.load(f)
        with open(os.path.join(model_dir, "metrics.json")) as f:
            metrics = json.load(f)
    except Exception as error:
        results.append(("The files load", False, f"{type(error).__name__}: {str(error)[:100]}"))
        finish(results)
    results.append(("The files load", True, "model, scaler, config and metrics"))

    # 3. shapes, 4. beats the baseline, 5. average improvement
    results.append(("Shapes match config.json", *check_shapes(model, scaler, config)))
    results.append(("Beats the baseline every day", *check_beats_baseline_every_day(metrics)))
    results.append(("At least 5% better on average", *check_average_improvement(metrics)))

    # 6. predictions, 7. fresh data (both need the CSV data)
    try:
        df = load_raw(cfg)
        results.append(("Predictions are believable", *check_predictions(model, scaler, config, df, cfg)))
        results.append(("The data is fresh", *check_fresh_data(df)))
    except Exception as error:
        results.append(("Data and predictions", False, f"{type(error).__name__}: {str(error)[:100]}"))

    finish(results)


if __name__ == "__main__":
    main()
