from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def train_models(df):

    features = [
        'payments_started', 'payments_ended',
        'lag1', 'lag2', 'lag3',
        'hour', 'is_peak'
    ]

    X = df[features]
    y = df['true_occupancy']

    # Time-series split (NO shuffle)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print(f"\n{name}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.4f}")