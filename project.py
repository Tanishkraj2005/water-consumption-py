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

# Fix: Revenue Month parsing
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


# Objective 4: ML Prediction
monthly_df = monthly_consumption.reset_index()
monthly_df.columns = ['Month', 'Consumption']
monthly_df['Timestamp'] = monthly_df['Month'].astype(np.int64) // 10**9

X = monthly_df[['Timestamp']]
y = monthly_df['Consumption']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

monthly_df['Predicted'] = model.predict(X)

plt.figure(figsize=(14, 6))
plt.plot(monthly_df['Month'], monthly_df['Consumption'], label='Actual')
plt.plot(monthly_df['Month'], monthly_df['Predicted'], label='Predicted', linestyle='--')
plt.title("Actual vs. Predicted Monthly Consumption")
plt.ylabel("Consumption (HCF)")
plt.xlabel("Date")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("=== OBJECTIVE 4 ===")
next_month_timestamp = monthly_df['Timestamp'].max() + 2629743
next_prediction = model.predict([[next_month_timestamp]])
print(f"Predicted Next Month Consumption: {next_prediction[0]:,.2f} HCF\n")


# Objective 5: Correlation by ZIP
def safe_corr(group):
    if len(group) > 1:
        return pearsonr(group['Consumption (HCF)'], group['Current Charges'])[0]
    return np.nan

zip_corr = df.groupby('RC Code').apply(safe_corr).dropna()
zip_corr_sorted = zip_corr.sort_values()

plt.figure(figsize=(12, 6))
zip_corr_sorted.plot(kind='bar', color='skyblue')
plt.title("Correlation Between Consumption and Cost by ZIP Code")
plt.ylabel("Pearson Correlation Coefficient")
plt.xlabel("ZIP Code (RC Code)")
plt.tight_layout()
plt.show()

print("=== OBJECTIVE 5 ===")
overall_corr, _ = pearsonr(df['Consumption (HCF)'], df['Current Charges'])
print(f"Overall Correlation: {overall_corr:.2f}")
print("Correlation plot by ZIP saved.\n")
