import numpy as np
from sklearn.metrics import root_mean_squared_error


def to_celsius(x, scaler):
    """Turn scaled temperatures (0 to 1) back into degrees Celsius."""
    t_min = scaler.data_min_[0]
    t_max = scaler.data_max_[0]
    return x * (t_max - t_min) + t_min


def evaluate(model, X_test, y_test, scaler, baseline_pred):
    """Compare the LSTM with the baseline on the test set, one day at a time."""
    actual = to_celsius(y_test, scaler)
    baseline = to_celsius(baseline_pred, scaler)
    lstm = to_celsius(model.predict(X_test, verbose=0), scaler)

    baseline_rmse = []
    lstm_rmse = []
    print("Day | Baseline | LSTM | Better?")
    for day in range(actual.shape[1]):
        b = root_mean_squared_error(actual[:, day], baseline[:, day])
        l = root_mean_squared_error(actual[:, day], lstm[:, day])
        baseline_rmse.append(b)
        lstm_rmse.append(l)
        print(f" {day + 1}  |  {b:.2f}  | {l:.2f} | {'yes' if l < b else 'NO'}")

    metrics = {
        "baseline_rmse": [round(float(x), 3) for x in baseline_rmse],
        "lstm_rmse": [round(float(x), 3) for x in lstm_rmse],
        "baseline_avg": round(float(np.mean(baseline_rmse)), 3),
        "lstm_avg": round(float(np.mean(lstm_rmse)), 3),
    }
    print(f"\nAverage: baseline {metrics['baseline_avg']} C, LSTM {metrics['lstm_avg']} C")
    return metrics
