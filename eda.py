import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/supply_chain_shipments.csv")

print("Dataset Loaded Successfully")
print(df.head())
print(df.columns)

# Numeric features
numeric_cols = [
    "supplier_rating",
    "distance_km",
    "weather_score",
    "demand_volatility",
    "inventory_level",
    "order_quantity",
    "risk_score"
]

print("Starting Boxplots...")

for col in numeric_cols:
    plt.figure(figsize=(6,4))
    sns.boxplot(x="is_delayed", y=col, data=df)
    plt.title(f"{col} vs Delay")
    plt.show()

print("Boxplots Finished")
