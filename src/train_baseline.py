import numpy as np


def baseline_predict(X, horizon):
    """Baseline: every future day is the same as the last known day."""
    last_temp = X[:, -1, 0]                                   # last day's temp_max of every window
    return np.repeat(last_temp.reshape(-1, 1), horizon, axis=1)

