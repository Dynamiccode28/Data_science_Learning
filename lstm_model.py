import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def prepare_lstm_data(df, feature_cols, target_col, window=10):

    data = df[feature_cols + [target_col]].values

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []

    for i in range(len(data_scaled) - window):
        X.append(data_scaled[i:i+window, :-1])   # features
        y.append(data_scaled[i+window, -1])      # target

    X = np.array(X)
    y = np.array(y)

    return X, y, scaler


def build_lstm(input_shape):

    model = Sequential()

    model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    return model