import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import os

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    print("="*50)
    print(" LAB 6: NN on Smoothed COVID-19 Data (Forecasting)")
    print("="*50)

    # 1. Load Data
    dataset_path = "../dataset/covid/owid-covid-data.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "dataset/covid/owid-covid-data.csv"
    
    print(f"Loading dataset from: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print("Error: Dataset not found.")
        return

    # 2. Filter Data (Thailand, Smoothed Cases)
    country = 'Thailand'
    target_col = 'new_cases_smoothed'
    
    print(f"Filtering data for: {country}")
    df_thai = df[df['location'] == country].copy()
    df_thai['date'] = pd.to_datetime(df_thai['date'])
    df_thai = df_thai.sort_values('date')
    
    # Handle Missing Values
    df_thai[target_col] = df_thai[target_col].fillna(0)
    
    data = df_thai[target_col].values.reshape(-1, 1)
    dates = df_thai['date'].values
    
    print(f"Total Records: {len(data)}")

    # 3. Preprocessing
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Create Sequences
    seq_length = 30 # Look back 30 days to predict next day
    X, y = create_sequences(data_normalized, seq_length)
    
    # Split Train/Test (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Flatten input for MLP (samples, features) -> (samples, seq_length)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Testing Samples: {X_test.shape[0]}")

    # 4. Train NN Model
    # 10 Hidden Layers, 100 Nodes each
    hidden_layers = tuple([100] * 10)
    print(f"Training MLPRegressor with Architecture: {hidden_layers}")
    
    mlp = MLPRegressor(hidden_layer_sizes=hidden_layers, 
                       activation='relu', 
                       solver='adam', 
                       max_iter=500, 
                       random_state=42)
    
    mlp.fit(X_train, y_train.ravel())
    
    # 5. Evaluate
    y_pred_test = mlp.predict(X_test)
    y_pred_test_inv = scaler.inverse_transform(y_pred_test.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    score = r2_score(y_test_inv, y_pred_test_inv)
    print(f"Model R2 Score (Accuracy): {score:.4f}")

    # 6. Forecast Future
    # We need the last sequence from the entire data to start forecasting future
    last_sequence = data_normalized[-seq_length:].reshape(1, -1)
    
    future_days_3m = 90
    future_days_6m = 180
    
    forecast_3m = []
    curr_seq = last_sequence.copy()
    
    print(f"Forecasting next {future_days_6m} days...")
    
    for i in range(future_days_6m):
        next_pred = mlp.predict(curr_seq)[0]
        
        if i < future_days_3m:
            forecast_3m.append(next_pred)
            
        # Update sequence: remove first element, add new prediction
        curr_seq = np.append(curr_seq[:, 1:], [[next_pred]], axis=1)

    forecast_6m = forecast_3m + [curr_seq[0, -1]] # Continuation (simplified logic for list storage)
    # Actually, let's re-run loop or just store all 180
    
    # Re-doing clean forecast list
    forecast_full = []
    curr_seq = last_sequence.copy()
    for i in range(future_days_6m):
        next_pred = mlp.predict(curr_seq)[0]
        forecast_full.append(next_pred)
        curr_seq = np.append(curr_seq[:, 1:], [[next_pred]], axis=1)
        
    forecast_full_inv = scaler.inverse_transform(np.array(forecast_full).reshape(-1, 1))
    
    # Generate Dates for Forecast
    last_date = dates[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days_6m)

    # 7. Plot Results
    plt.figure(figsize=(12, 6))
    
    # Plot Actual Data (Historical)
    plt.plot(dates, data, label='Actual / Smoothed Trend', color='blue', alpha=0.6)
    
    # Plot Test Predictions (to see model fit)
    # Align dates for test predictions
    test_dates = dates[train_size+seq_length+1 : train_size+seq_length+1+len(y_test)]
    # Note: alignment can be tricky with sequences. 
    # Index of y_test[0] corresponds to original data index: train_size + seq_length
    test_date_indices = range(train_size + seq_length, train_size + seq_length + len(y_test))
    test_dates_plot = dates[test_date_indices]
    
    plt.plot(test_dates_plot, y_pred_test_inv, label='Model Test Prediction', color='orange', linestyle='--')
    
    # Plot 3-Month Forecast
    plt.plot(future_dates[:90], forecast_full_inv[:90], label='3-Month Forecast', color='red', linewidth=2)
    
    # Plot 6-Month Forecast (Extension)
    plt.plot(future_dates[90:], forecast_full_inv[90:], label='6-Month Forecast', color='purple', linestyle=':', linewidth=2)

    plt.title(f'COVID-19 Forecasting (Thailand) - R2 Score: {score:.4f}')
    plt.xlabel('Date')
    plt.ylabel('New Cases Smoothed')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
