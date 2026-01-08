import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# ตั้งค่า GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_data():
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, '..', 'dataset', 'covid', 'owid-covid-data.csv')
    
    if not os.path.exists(file_path):
        # Fallback
        file_path = 'dataset/covid/owid-covid-data.csv'
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return None

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Filter for Thailand
    df_tha = df[df['iso_code'] == 'THA'].copy()
    
    # 2. Select target: new_cases_smoothed
    # Fill NaN with 0 for early days
    df_tha['new_cases_smoothed'] = df_tha['new_cases_smoothed'].fillna(0)
    
    # Sort by date
    df_tha['date'] = pd.to_datetime(df_tha['date'])
    df_tha = df_tha.sort_values('date')
    
    data = df_tha['new_cases_smoothed'].values.reshape(-1, 1)
    dates = df_tha['date'].values
    
    print(f"Loaded Thailand data: {len(data)} days.")
    return data, dates

def create_sequences(data, lookback_window=30):
    X, y = [], []
    for i in range(len(data) - lookback_window):
        X.append(data[i:i+lookback_window])
        y.append(data[i+lookback_window])
    return np.array(X), np.array(y)

def build_cnn1d_regression(num_layers, num_nodes, input_shape):
    model = models.Sequential()
    
    for i in range(num_layers):
        if i == 0:
             model.add(layers.Conv1D(num_nodes, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
        else:
             model.add(layers.Conv1D(num_nodes, kernel_size=3, activation='relu', padding='same'))
        
        if i % 2 == 0:
             model.add(layers.MaxPooling1D(pool_size=2, padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='linear')) # Regression
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def calculate_accuracy(y_true, y_pred):
    # Custom "Accuracy" for regression: 1 - (MAE / Mean)
    mae = mean_absolute_error(y_true, y_pred)
    mean_val = np.mean(y_true)
    if mean_val == 0: return 0
    accuracy = (1 - (mae / mean_val)) * 100
    return max(0, accuracy)

def main():
    # 1. Load Data
    data, dates = load_data()
    if data is None: return

    # 2. Preprocess
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    LOOKBACK = 30
    X, y = create_sequences(data_scaled, LOOKBACK)
    
    # Split Train/Test (Time series split, no shuffle)
    split_size = int(len(X) * 0.8)
    X_train, y_train = X[:split_size], y[:split_size]
    X_test, y_test = X[split_size:], y[split_size:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    input_shape = (LOOKBACK, 1)

    # --- Experiment: Network Sizes ---
    print("\n" + "="*50)
    print(" Experiment: Network Sizes (Layers & Nodes)")
    print("="*50)
    
    layers_to_test = [1, 2, 3]
    nodes_to_test = [32, 64, 128]
    results = []
    
    best_model = None
    best_acc = 0
    
    for l in layers_to_test:
        for n in nodes_to_test:
            print(f"Testing: {l} Layers, {n} Nodes...")
            model = build_cnn1d_regression(l, n, input_shape)
            
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
            
            loss, mae = model.evaluate(X_test, y_test, verbose=0)
            
            # Predict for accuracy calculation
            pred_scaled = model.predict(X_test, verbose=0)
            pred = scaler.inverse_transform(pred_scaled)
            actual = scaler.inverse_transform(y_test)
            
            acc = calculate_accuracy(actual, pred)
            
            results.append({'Layers': l, 'Nodes': n, 'Accuracy (%)': acc})
            
            if acc > best_acc:
                best_acc = acc
                best_model = model

    df_results = pd.DataFrame(results)
    pivot_table = df_results.pivot(index='Layers', columns='Nodes', values='Accuracy (%)')
    print("\nResults Table (Forecast Accuracy %):")
    print(pivot_table)
    print("="*50)
    
    # --- Forecasting (3 & 6 Months) ---
    if best_model:
        print("\nGenerating Forecasts...")
        
        # Helper to forecast n steps
        def forecast_future(model, last_sequence, n_steps):
            future_forecast = []
            curr_seq = last_sequence.copy()
            
            for _ in range(n_steps):
                # Predict next point
                # curr_seq shape (LOOKBACK, 1) -> (1, LOOKBACK, 1)
                next_pred = model.predict(curr_seq.reshape(1, LOOKBACK, 1), verbose=0)[0, 0]
                future_forecast.append(next_pred)
                
                # Update sequence: remove first, add pred
                curr_seq = np.roll(curr_seq, -1)
                curr_seq[-1] = next_pred
                
            return np.array(future_forecast).reshape(-1, 1)

        # Use the very last data sequence to predict into unknown future
        last_seq = data_scaled[-LOOKBACK:]
        
        # 3 Months ~ 90 Days
        forecast_3m = forecast_future(best_model, last_seq, 90)
        forecast_3m = scaler.inverse_transform(forecast_3m)
        
        # 6 Months ~ 180 Days
        forecast_6m = forecast_future(best_model, last_seq, 180)
        forecast_6m = scaler.inverse_transform(forecast_6m)

        # Plotting
        plt.figure(figsize=(12, 6))
        
        # Plot only last 365 days of actual data for clarity
        plt.plot(dates[-365:], scaler.inverse_transform(data_scaled)[-365:], label='Actual (Smoothed)', color='blue')
        
        # Generate future dates
        last_date = dates[-1]
        dates_3m = pd.date_range(start=last_date, periods=91)[1:]
        dates_6m = pd.date_range(start=last_date, periods=181)[1:]
        
        plt.plot(dates_6m, forecast_6m, label='6-Month Forecast', color='red', linestyle='--')
        plt.plot(dates_3m, forecast_3m, label='3-Month Forecast', color='orange', linestyle='--')
        
        plt.title('COVID-19 Thailand: Smoothed Cases vs CNN Forecast')
        plt.xlabel('Date')
        plt.ylabel('New Cases Smoothed')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
