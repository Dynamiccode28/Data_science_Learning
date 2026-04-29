import numpy as np
import pandas as pd

def generate_parking_data(n_days=60, freq='5min', seed=42):
    np.random.seed(seed)

    # Time index
    time_index = pd.date_range(
        start='2024-01-01',
        periods=n_days * 24 * 12,
        freq=freq
    )

    df = pd.DataFrame({'time': time_index})
    df['hour'] = df['time'].dt.hour

    # Traffic demand pattern (morning + evening peaks)
    def traffic_pattern(hour):
        return (
            12 * np.exp(-0.5 * ((hour - 9) / 2) ** 2) +   # morning peak
            10 * np.exp(-0.5 * ((hour - 18) / 2) ** 2)    # evening peak
        )

    df['base_demand'] = df['hour'].apply(traffic_pattern)

    # Generate arrivals
    df['payments_started'] = np.random.poisson(df['base_demand'])

    # Initialize columns
    active = 0
    active_list = []
    ended_list = []

    departure_rate = 0.12  # % of cars leaving per timestep

    # Sequential simulation (CRITICAL FIX)
    for i in range(len(df)):
        started = df.loc[i, 'payments_started']

        # departures depend on CURRENT active cars
        ended = int(active * departure_rate)

        # update active vehicles
        active = active + started - ended
        active = max(active, 0)

        active_list.append(active)
        ended_list.append(ended)

    df['payments_ended'] = ended_list
    df['active_payments'] = active_list

    # Add realistic noise (unpaid parking etc.)
    noise = np.random.randint(0, 4, len(df))
    df['true_occupancy'] = df['active_payments'] + noise

    return df


if __name__ == "__main__":
    df = generate_parking_data()

    df.to_csv("parking_data.csv", index=False)

    print("Data Generated Successfully")
    print("\nHEAD:")
    print(df.head())

    print("\nSTATS:")
    print(df.describe())