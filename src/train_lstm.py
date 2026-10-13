import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"        # hide TensorFlow's startup messages

import json
import sys

import joblib
import yaml
from tensorflow import keras

from src.data_loader import check_data, load_raw, split_by_city
from src.evaluate import evaluate
from src.feature_engineering import make_windows, scale
from src.train_baseline import baseline_predict
import glob
import hashlib
import subprocess

import mlflow


def build_model(cfg):

    units = cfg["model"]["lstm_units"]
    dropout = cfg["model"]["dropout"]

    model = keras.Sequential([
        keras.layers.Input(shape=(cfg["window"], len(cfg["features"]))),
        keras.layers.LSTM(units[0], return_sequences=True),
        keras.layers.Dropout(dropout),
        keras.layers.LSTM(units[1]),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(cfg["horizon"]),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["model"]["learning_rate"]),
        loss=cfg["model"]["loss"],
    )
    return model


def log_run(cfg, run_name, metrics, history, out_dir):
    """Save this training run in MLflow (the mlruns/ folder): settings, scores, loss curve, model files."""
    raw_dir = cfg["data"]["raw_dir"]

    hasher = hashlib.sha256()                  # one short code that changes if the data changes
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*_weather.csv")))
    for path in csv_files:
        with open(path, "rb") as f:
            hasher.update(f.read())

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "unknown"

    improvement = (1 - metrics["lstm_avg"] / metrics["baseline_avg"]) * 100
    beats_all_days = all(l < b for l, b in zip(metrics["lstm_rmse"], metrics["baseline_rmse"]))

    # mlflow.set_tracking_uri("URI of remote mlflow setup")
    mlflow.set_experiment("weather-lstm")
    
    with mlflow.start_run(run_name=run_name):
        
        mlflow.log_params({
            "window": cfg["window"], "horizon": cfg["horizon"],
            "lstm_units": str(cfg["model"]["lstm_units"]), "dropout": cfg["model"]["dropout"],
            "learning_rate": cfg["model"]["learning_rate"], "epochs": cfg["training"]["epochs"],
            "batch_size": cfg["training"]["batch_size"], "seed": cfg["training"]["seed"],
            "data_hash": hasher.hexdigest()[:12], "data_files": len(csv_files),
        })
        mlflow.set_tag("git_commit", commit)

        # the loss curve: one point per epoch
        for epoch, (train_loss, val_loss) in enumerate(zip(history.history["loss"], history.history["val_loss"])):
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

        # the scores used to decide which model is better
        for day, (b, l) in enumerate(zip(metrics["baseline_rmse"], metrics["lstm_rmse"]), start=1):
            mlflow.log_metric(f"baseline_rmse_day{day}", b)
            mlflow.log_metric(f"lstm_rmse_day{day}", l)
        mlflow.log_metrics({
            "baseline_avg": metrics["baseline_avg"],
            "lstm_avg": metrics["lstm_avg"],
            "improvement_pct": improvement,
            "beats_baseline_all_days": int(beats_all_days),
            "epochs_run": len(history.history["loss"]),
        })

        # this run's own copy of the model, scaler, config and metrics
        mlflow.log_artifacts(out_dir, artifact_path="files")

def main():
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/train.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    features = cfg["features"]
    window = cfg["window"]
    horizon = cfg["horizon"]
    out_dir = cfg["output"]["dir"]
    os.makedirs(out_dir, exist_ok=True)

    assert cfg["target"] == features[0], "the target must be the first feature"
    keras.utils.set_random_seed(cfg["training"]["seed"])

    # 1. data
    df = load_raw(cfg)
    print("Combined shape:", df.shape)
    check_data(df)
    train_df, val_df, test_df = split_by_city(df, cfg)
    print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))

    # 2. features
    train_df, val_df, test_df, scaler = scale(train_df, val_df, test_df, features)

    X_train, y_train = make_windows(train_df, features, window, horizon)
    
    X_val, y_val = make_windows(val_df, features, window, horizon)
    
    
    X_test, y_test = make_windows(test_df, features, window, horizon)
    
    print("Windows -> Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    # 3. train
    model = build_model(cfg)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg["training"]["early_stopping_patience"], restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=cfg["training"]["reduce_lr_factor"], patience=cfg["training"]["reduce_lr_patience"]
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(out_dir, "best_lstm.keras"), monitor="val_loss", save_best_only=True
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        callbacks=callbacks,
        verbose=2,
    )

    # 4. evaluate against the baseline
    baseline_pred = baseline_predict(X_test, horizon)
    metrics = evaluate(model, X_test, y_test, scaler, baseline_pred)

    # 5. save the files the forecast service needs
    model.save(os.path.join(out_dir, "lstm_weather_model.keras"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"features": features, "target": cfg["target"], "window": window, "horizon": horizon}, f, indent=2)
        
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved to", out_dir, ":", sorted(os.listdir(out_dir)))
    
    
    # 6. record this run in MLflow (folder: mlruns/)
    run_name = os.path.splitext(os.path.basename(config_path))[0]
    log_run(cfg, run_name, metrics, history, out_dir)


if __name__ == "__main__":
    main()
