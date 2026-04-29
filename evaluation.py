import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_lstm(model, X_test, y_test, scaler, n_features):

    preds = model.predict(X_test)

    # Prepare arrays for inverse scaling
    temp_pred = np.zeros((len(preds), n_features + 1))
    temp_true = np.zeros((len(y_test), n_features + 1))

    temp_pred[:, -1] = preds.flatten()
    temp_true[:, -1] = y_test.flatten()

    # Inverse transform
    preds_inv = scaler.inverse_transform(temp_pred)[:, -1]
    y_test_inv = scaler.inverse_transform(temp_true)[:, -1]

    rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))
    r2 = r2_score(y_test_inv, preds_inv)

    print("\nLSTM Results (Original Scale)")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    return preds_inv, y_test_inv


# ✅ NEW FUNCTION: Convert predictions → real-world interpretation
def parking_status(preds, capacity=60):

    results = []

    for occ in preds:

        occ = max(0, occ)  # safety

        available = capacity - occ
        available = max(0, available)

        occupancy_rate = occ / capacity

        # Status logic
        if occupancy_rate < 0.7:
            status = "Available"
        elif occupancy_rate < 0.9:
            status = "Limited"
        else:
            status = "Full"

        results.append({
            "predicted_occupancy": round(occ, 2),
            "available_spots": round(available, 2),
            "occupancy_rate": round(occupancy_rate * 100, 2),
            "status": status
        })

    return results