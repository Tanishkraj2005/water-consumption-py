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


# Objective 2: Cost Over Time
monthly_cost = df.groupby(df['Revenue Month'].dt.to_period('M'))['Current Charges'].sum()
monthly_cost.index = monthly_cost.index.to_timestamp()

plt.figure(figsize=(14, 6))
monthly_cost.plot(color='orange')
plt.title("Monthly Total Water Cost")
plt.ylabel("Total Charges ($)")
plt.xlabel("Date")
plt.grid(True)
plt.tight_layout()
plt.show()

print("=== OBJECTIVE 2 ===")
print("Water cost trend graph saved as 'graph_objective2_cost_trend.png'\n")


# Objective 3: Per Capita ZIP Use
zip_consumption = df.groupby('RC Code')['Consumption (HCF)'].sum()
zip_counts = df['RC Code'].value_counts()
per_capita_zip = zip_consumption / zip_counts
per_capita_zip_sorted = per_capita_zip.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
per_capita_zip_sorted.head(10).plot(kind='bar', color='green')
plt.title("Top 10 ZIP Codes by Per Capita Consumption")
plt.ylabel("Per Capita Consumption (HCF)")
plt.xlabel("ZIP Code (RC Code)")
plt.tight_layout()

plt.figure(figsize=(12, 6))
per_capita_zip_sorted.tail(10).plot(kind='bar', color='purple')
plt.title("Bottom 10 ZIP Codes by Per Capita Consumption")
plt.ylabel("Per Capita Consumption (HCF)")
plt.xlabel("ZIP Code (RC Code)")
plt.tight_layout()
plt.show()

print("=== OBJECTIVE 3 ===")
print("Top and bottom 10 ZIP per capita graphs saved.\n")
