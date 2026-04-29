from data_simulation import generate_parking_data
from feature_engineering import create_features
from model_training import train_models
from evaluation import evaluate_lstm,parking_status
from lstm_model import prepare_lstm_data, build_lstm
# Step 1: Generate data
df = generate_parking_data(n_days=60)

print("\nRaw Shape:", df.shape)

# Step 2: Feature Engineering
df_fe = create_features(df)

print("\nFeature Engineered Shape:", df_fe.shape)
print("\nColumns:")
print(df_fe.columns)

print("\nSample:")
print(df_fe.head())
print("Training Models...")
train_models(df_fe)


print("\nPreparing LSTM data...")

feature_cols = [
    'payments_started',
    'payments_ended',
    'lag1', 'lag2', 'lag3',
    'hour', 'is_peak'
]

target_col = 'true_occupancy'

X, y, scaler = prepare_lstm_data(df_fe, feature_cols, target_col)

split = int(0.8 * len(X))

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("LSTM Input Shape:", X_train.shape)

model = build_lstm((X_train.shape[1], X_train.shape[2]))

print("\nTraining LSTM...")
model.fit(X_train, y_train, epochs=10, batch_size=32,
          validation_data=(X_test, y_test))




# Evaluate LSTM
print("\nEvaluating LSTM...")
n_features = len(feature_cols)

preds, y_actual = evaluate_lstm(
    model,
    X_test,
    y_test,
    scaler,
    n_features
)

# ✅ Convert to real-world parking insights
status_results = parking_status(preds, capacity=60)

print("\nSample Parking Insights:\n")

for i in range(10):
    res = status_results[i]
    print(
        f"Occupancy: {res['predicted_occupancy']} | "
        f"Available: {res['available_spots']} | "
        f"Usage: {res['occupancy_rate']}% | "
        f"Status: {res['status']}"
    )