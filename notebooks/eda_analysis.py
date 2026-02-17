import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/supply_chain_shipments.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nDelay Distribution:")
print(df["is_delayed"].value_counts(normalize=True))


# -----------------------------
# 1. Delay by Supplier Rating
# -----------------------------
plt.figure()
sns.barplot(data=df, x="supplier_rating", y="is_delayed")
plt.title("Delay Rate by Supplier Rating")
plt.show()


# -----------------------------
# 2. Delay by Shipment Mode
# -----------------------------
plt.figure()
sns.barplot(data=df, x="shipment_mode", y="is_delayed")
plt.title("Delay Rate by Shipment Mode")
plt.show()


# -----------------------------
# 3. Distance vs Delay
# -----------------------------
plt.figure()
sns.boxplot(data=df, x="is_delayed", y="distance_km")
plt.title("Distance Distribution by Delay Status")
plt.show()


# -----------------------------
# 4. Weather Impact
# -----------------------------
plt.figure()
sns.boxplot(data=df, x="is_delayed", y="weather_score")
plt.title("Weather Impact on Delay")
plt.show()
