import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

print("1. Configuration & Data Loading...")
downloads_path = r"C:\Users\YEDHEE\Downloads"
csv_file_path = None

print(f"Searching for CSV file in {downloads_path}...")
for file in os.listdir(downloads_path):
    if "covid" in file.lower() and file.endswith(".csv"):
        csv_file_path = os.path.join(downloads_path, file)
        print(f" -> Found potential file: {file}")
        break

if csv_file_path is None:
    print("[Error] No COVID-19 CSV file found in Downloads.")
    exit()

print(f"Reading CSV from: {csv_file_path}")
df = pd.read_csv(csv_file_path)

print("Filtering data for Thailand (THA)...")
country = 'Thailand'
df = df[df['location'] == country].copy()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Use 'new_cases_smoothed' and fill NaN with 0
series = df['new_cases_smoothed'].fillna(0).values
dates = df['date'].values

print("2. Creating Feature from Past (Sliding Window)...")
window = 14 # Look back 14 days
X = []
y = []

# Sliding window creation loop
for i in range(window, len(series)):
    X.append(series[i-window:i])
    y.append(series[i])

X = np.array(X)
y = np.array(y)
target_dates = dates[window:]

print(f"Total samples created: {len(X)}")

print("3. Splitting Train/Test (Last 90 days for Test)...")
test_horizon = 90 # 3 Months
X_train = X[:-test_horizon]
y_train = y[:-test_horizon]
X_test = X[-test_horizon:]
y_test = y[-test_horizon:]

dates_train = target_dates[:-test_horizon]
dates_test = target_dates[-test_horizon:]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

print("Scaling Features...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

kernels = ['linear', 'poly', 'rbf']
rmse_scores = {}
models = {}

print("4. Training SVR Models (Linear, Poly, RBF)...")
for k in kernels:
    print(f" -> Training Kernel: {k} ...")
    model = SVR(kernel=k, C=10.0, gamma='scale')
    model.fit(X_train_s, y_train)
    
    y_pred = model.predict(X_test_s)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"    Kernel = {k:<6}: RMSE = {rmse:.2f}")
    rmse_scores[k] = rmse
    models[k] = model

print("5. 3-Month Forecasting (Recursive)...")
# Use RBF model (usually best) or the best one found
best_model = models['rbf'] 

last_window = series[-window:].copy()
future_steps = 90 # 90 days ahead
future_preds = []

print("Generating future predictions...")
# Iterate to predict next day, then add prediction to window, slide, and repeat
for _ in range(future_steps):
    # Scale current window
    x_win = scaler.transform(last_window.reshape(1, -1))
    next_val = best_model.predict(x_win)[0]
    
    future_preds.append(next_val)
    
    # Update window: remove first item, add new prediction
    last_window = np.roll(last_window, -1)
    last_window[-1] = next_val

future_preds = np.array(future_preds)
future_dates = pd.date_range(df['date'].iloc[-1] + pd.Timedelta(days=1), periods=future_steps, freq='D')

print("6. Plotting Results...")
plt.figure(figsize=(12, 6))

# Plot Actual History
plt.plot(dates, series, color='black', label='Actual (Smoothed)')
# Plot Test Interval (Actual)
plt.plot(dates_test, y_test, color='blue', label='Actual (Test)', alpha=0.5)

# Plot Forecast
plt.plot(future_dates, future_preds, color='orange', label='SVR Forecast (Next 3 months)')

plt.title(f'SVR Forecasting of Smoothed COVID-19 Cases - {country}')
plt.xlabel('Date')
plt.ylabel('New Cases (Smoothed)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nResult Summary:")
for k, r in rmse_scores.items():
    print(f"Kernel = {k:<6}: RMSE = {r:.2f}")
