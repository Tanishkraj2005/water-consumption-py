import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# Base CSV load and preprocessing
fp = "D:/Python Project/Water_Consumption_And_Cost__2013_-_Feb_2025_.csv"
df = pd.read_csv(fp)

# Fix Revenue Month parsing
df['Revenue Month'] = pd.to_datetime(df['Revenue Month'], errors='coerce')
df = df.dropna(subset=['Revenue Month', 'Consumption (HCF)', 'Current Charges'])
df['Year'] = df['Revenue Month'].dt.year
df['Month'] = df['Revenue Month'].dt.month


# Objective 1: Highest Consumption
monthly_consumption = df.groupby(df['Revenue Month'].dt.to_period('M'))['Consumption (HCF)'].sum()
monthly_consumption.index = monthly_consumption.index.to_timestamp()

max_month = monthly_consumption.idxmax()
max_value = monthly_consumption.max()
avg_value = monthly_consumption.mean()

plt.figure(figsize=(14, 6))
monthly_consumption.plot()
plt.axhline(avg_value, color='red', linestyle='--', label='Average Consumption')
plt.title("Monthly Total Water Consumption")
plt.ylabel("Consumption (HCF)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("=== OBJECTIVE 1 ===")
print(f"Highest Consumption Month: {max_month}")
print(f"Max Consumption: {max_value:,.2f} HCF")
print(f"Average Monthly: {avg_value:,.2f} HCF\n")
