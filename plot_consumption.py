#!/usr/bin/env python3
"""
plot_consumption.py

Standalone script based on `test.ipynb` that:
- generates synthetic electricity consumption data
- filters out extreme values
- trains a simple LinearRegression model
- creates and saves charts (histogram, scatter+regression, boxplot by hour, correlation heatmap)

Run: python3 plot_consumption.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Settings
OUTPUT_DIR = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)
N = 1000  # Number of hourly data points

# --- Generate synthetic data (from test.ipynb) ---
outdoor_temp = np.random.uniform(10, 35, N)
time_of_day = np.random.randint(0, 24, N)
occupants = np.random.randint(1, 6, N)
thermostat_set = np.random.uniform(19, 24, N)

consumption = (
    0.5 * outdoor_temp
    + 0.1 * time_of_day
    + 0.8 * occupants
    - 0.2 * thermostat_set
    + 5  # bias/baseline
    + np.random.normal(0, 2, N)  # noise
)
consumption = np.maximum(0, consumption)

data = pd.DataFrame({
    'OutdoorTemp': outdoor_temp,
    'TimeOfDay': time_of_day,
    'Occupants': occupants,
    'ThermostatSet': thermostat_set,
    'Consumption_kWh': consumption,
})

# --- Basic filtering / outlier handling ---
upper_bound = data['Consumption_kWh'].quantile(0.99)
data_filtered = data[data['Consumption_kWh'] < upper_bound].copy()
print(f"Data points before filtering: {len(data)}")
print(f"Data points after filtering (Consumption < {upper_bound:.2f} kWh): {len(data_filtered)}")

# --- Train a simple model (optional) ---
X = data_filtered[['OutdoorTemp', 'TimeOfDay', 'Occupants', 'ThermostatSet']]
y = data_filtered['Consumption_kWh']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Model metrics on test set:")
print(f"  MSE: {mean_squared_error(y_test, y_pred):.3f}")
print(f"  R2 : {r2_score(y_test, y_pred):.3f}")

# --- Plot 1: Histogram of consumption ---
plt.figure(figsize=(8, 5))
sns.histplot(data_filtered['Consumption_kWh'], bins=30, kde=True)
plt.title('Consumption (kWh) Distribution')
plt.xlabel('Consumption (kWh)')
plt.tight_layout()
fn = os.path.join(OUTPUT_DIR, 'consumption_histogram.png')
plt.savefig(fn)
print(f"Saved histogram to {fn}")

# --- Plot 2: Scatter OutdoorTemp vs Consumption with regression line ---
plt.figure(figsize=(8, 5))
sns.regplot(x='OutdoorTemp', y='Consumption_kWh', data=data_filtered, scatter_kws={'s': 20, 'alpha': 0.6})
plt.title('Outdoor Temperature vs Consumption')
plt.xlabel('Outdoor Temperature (°C)')
plt.ylabel('Consumption (kWh)')
plt.tight_layout()
fn = os.path.join(OUTPUT_DIR, 'temp_vs_consumption.png')
plt.savefig(fn)
print(f"Saved scatter/regression to {fn}")

# --- Plot 3: Boxplot of consumption by hour of day ---
plt.figure(figsize=(10, 6))
sns.boxplot(x='TimeOfDay', y='Consumption_kWh', data=data_filtered)
plt.title('Consumption by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Consumption (kWh)')
plt.tight_layout()
fn = os.path.join(OUTPUT_DIR, 'consumption_by_hour_boxplot.png')
plt.savefig(fn)
print(f"Saved boxplot to {fn}")

# --- Plot 4: Correlation heatmap ---
plt.figure(figsize=(6, 5))
corr = data_filtered[['OutdoorTemp', 'TimeOfDay', 'Occupants', 'ThermostatSet', 'Consumption_kWh']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation')
plt.tight_layout()
fn = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
plt.savefig(fn)
print(f"Saved heatmap to {fn}")

# Show plots (useful when running locally)
plt.show()

# Optionally, if the real dataset `energydata_complete.csv` exists, show top-level info
csv_path = 'energydata_complete.csv'
if os.path.exists(csv_path):
    print('\nFound `energydata_complete.csv` in workspace. Showing a quick preview:')
    try:
        df_real = pd.read_csv(csv_path, nrows=5)
        print(df_real.head())
    except Exception as e:
        print('Failed to read real CSV:', e)
else:
    print('\nNo `energydata_complete.csv` found or not present in working directory.')
