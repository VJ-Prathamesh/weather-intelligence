import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def to_celsius(x, scaler):
    """Turn scaled temp_max values (0 to 1) back into degrees Celsius."""
    t_min = scaler.data_min_[0]
    t_range = scaler.data_max_[0] - scaler.data_min_[0]
    return x * t_range + t_min


def rmse_per_day(actual, predicted):
    return [root_mean_squared_error(actual[:, day], predicted[:, day]) for day in range(actual.shape[1])]


def evaluate(model, X_test, y_test, scaler, baseline_pred):
    """Compare the LSTM with the baseline on the test set, day by day."""
    actual = to_celsius(y_test, scaler)
    baseline = to_celsius(baseline_pred, scaler)
    lstm = to_celsius(model.predict(X_test, verbose=0), scaler)

    baseline_rmse = rmse_per_day(actual, baseline)
    lstm_rmse = rmse_per_day(actual, lstm)

    print("Day | Baseline | LSTM  |  MAE  |  R2   | within 2C | Better?")
    for day in range(len(lstm_rmse)):
        mae = mean_absolute_error(actual[:, day], lstm[:, day])
        r2 = r2_score(actual[:, day], lstm[:, day])
        within_2 = (np.abs(actual[:, day] - lstm[:, day]) <= 2).mean() * 100
        better = "yes" if lstm_rmse[day] < baseline_rmse[day] else "NO"
        print(f" {day + 1}  |  {baseline_rmse[day]:.2f}    | {lstm_rmse[day]:.2f}  | {mae:.2f}  | {r2:.3f} |  {within_2:5.1f}%   | {better}")

    metrics = {
        "baseline_rmse": [round(float(x), 3) for x in baseline_rmse],
        "lstm_rmse": [round(float(x), 3) for x in lstm_rmse],
        "baseline_avg": round(float(np.mean(baseline_rmse)), 3),
        "lstm_avg": round(float(np.mean(lstm_rmse)), 3),
    }
    print(f"\nAverage -> Baseline: {metrics['baseline_avg']:.3f} C | LSTM: {metrics['lstm_avg']:.3f} C")
    return metrics
