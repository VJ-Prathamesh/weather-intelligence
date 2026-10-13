import numpy as np
from sklearn.preprocessing import MinMaxScaler


def scale(train_df, val_df, test_df, features):
    """Fit the scaler on the training data only, then scale all three sets with it."""
    scaler = MinMaxScaler()
    scaler.fit(train_df[features])

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df[features] = scaler.transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])
    test_df[features] = scaler.transform(test_df[features])
    return train_df, val_df, test_df, scaler


def make_windows(data_df, features, window, horizon):
    """Cut the data into windows: X = `window` days of all features, y = the next `horizon` days of temp_max."""
    X = []
    y = []
    for city in data_df["city"].unique():
        values = data_df[data_df["city"] == city][features].values   

        for i in range(len(values) - window - horizon + 1):
            X.append(values[i : i + window])                       
            y.append(values[i + window : i + window + horizon, 0]) 

    return np.array(X), np.array(y)
