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
