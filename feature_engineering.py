import pandas as pd

def create_features(df):

    df = df.copy()

    # Lag features (capture memory)
    df['lag1'] = df['true_occupancy'].shift(1)
    df['lag2'] = df['true_occupancy'].shift(2)
    df['lag3'] = df['true_occupancy'].shift(3)

    # Rolling features (trend + smoothing)
    df['rolling_mean_3'] = df['true_occupancy'].rolling(3).mean()
    df['rolling_std_3'] = df['true_occupancy'].rolling(3).std()

    # Peak hours
    df['is_peak'] = df['hour'].apply(
        lambda x: 1 if (8 <= x <= 10 or 17 <= x <= 19) else 0
    )

    # Drop NaN rows created by lagging
    df = df.dropna()

    return df