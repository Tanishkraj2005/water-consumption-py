# 💧 Water Consumption & Cost Analysis (2013–2025)

This project analyzes water consumption and cost data from 2013 to early 2025 using Python and data science tools. It answers key business and environmental questions through data processing, visualization, and predictive modeling.

---

## 📁 Dataset

- **Source:** `Water_Consumption_And_Cost__2013_-_Feb_2025_.csv`
- **Fields Used:**
  - `Revenue Month`
  - `Consumption (HCF)`
  - `Current Charges`
  - `RC Code` (ZIP/Area code)

---

## 📊 Objectives

### ✅ Objective 1: Highest Consumption Month
- Identifies the month with the highest total water usage.
- Compares each month’s usage against the overall average.
- 📈 Visualization: Time series plot with average line.

---

### 💵 Objective 2: Cost Over Time
- Tracks how water charges have changed monthly.
- 📈 Visualization: Monthly cost trend.

---

### 🧑‍🤝‍🧑 Objective 3: Per Capita ZIP Code Use
- Calculates per capita water use per ZIP code.
- 📊 Visualizes top and bottom 10 ZIPs by usage.

---

### 🤖 Objective 4: Machine Learning Forecast
- Uses linear regression to predict future water consumption.
- 📈 Visualization: Actual vs Predicted usage.

---

### 🧠 Objective 5: Correlation Analysis
- Calculates Pearson correlation between water usage and charges for each ZIP.
- 📊 Highlights areas with stronger or weaker cost-consumption relationships.

---

## 🛠️ Setup & Installation

1. Clone the repository or download the files.
2. Ensure you have Python 3.7+ installed.
3. Install required packages:
   ```bash
   pip install pandas matplotlib seaborn scikit-learn


## Run the full analysis:

- `python project.py`
