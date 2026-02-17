import numpy as np
import pandas as pd

np.random.seed(42)


# -----------------------------
# SUPPLIER GENERATION
# -----------------------------
def generate_suppliers(n_suppliers=50):
    supplier_ids = [f"S{i+1}" for i in range(n_suppliers)]

    ratings = np.random.choice(
        [1, 2, 3, 4, 5],
        size=n_suppliers,
        p=[0.05, 0.10, 0.40, 0.30, 0.15]
    )

    reliability_scores = np.clip(
        ratings / 5 + np.random.normal(0, 0.05, n_suppliers),
        0, 1
    )

    base_delay_prob = []
    for r in ratings:
        if r == 1:
            base_delay_prob.append(0.50)
        elif r == 2:
            base_delay_prob.append(0.35)
        elif r == 3:
            base_delay_prob.append(0.20)
        elif r == 4:
            base_delay_prob.append(0.10)
        else:
            base_delay_prob.append(0.05)

    regions = np.random.choice(
        ["Asia", "Europe", "North America"],
        size=n_suppliers
    )

    return pd.DataFrame({
        "supplier_id": supplier_ids,
        "supplier_rating": ratings,
        "reliability_score": reliability_scores,
        "base_delay_probability": base_delay_prob,
        "region": regions
    })


# -----------------------------
# SHIPMENT GENERATION
# -----------------------------
def generate_shipments(suppliers_df, n_shipments=30000):

    shipment_data = []
    shipment_modes = ["Air", "Sea", "Road"]

    for i in range(n_shipments):

        supplier = suppliers_df.sample(1).iloc[0]

        distance_km = np.random.randint(50, 2000)
        shipment_mode = np.random.choice(shipment_modes, p=[0.25, 0.35, 0.40])
        weather_score = np.random.uniform(0, 1)
        demand_volatility = np.random.uniform(0, 1)
        inventory_level = np.random.randint(0, 1000)
        order_quantity = np.random.randint(50, 500)

        # -------------------------
        # RISK CALCULATION (STRONG SIGNAL)
        # -------------------------

        # Base supplier risk
        risk = supplier["base_delay_probability"] * 0.5

        # Distance impact (stronger now)
        if distance_km > 1200:
            risk += 0.15
        elif distance_km > 800:
            risk += 0.08

        # Weather impact
        risk += 0.20 * weather_score

        # Demand volatility impact
        risk += 0.18 * demand_volatility

        # Inventory stress
        if inventory_level < order_quantity:
            risk += 0.15

        # Shipment mode impact
        if shipment_mode == "Sea":
            risk += 0.05
        elif shipment_mode == "Air" and weather_score > 0.6:
            risk += 0.12

        # Normalize risk
        risk = np.clip(risk, 0, 1)

        # Add small controlled noise
        noise = np.random.normal(0, 0.05)
        final_score = risk + noise

        # Strong threshold-based delay logic
        if final_score > 0.40:
            is_delayed = 1
        else:
            is_delayed = 0

        risk_score = int(risk * 100)

        shipment_data.append([
            f"O{i+1}",
            supplier["supplier_id"],
            supplier["supplier_rating"],
            distance_km,
            shipment_mode,
            weather_score,
            demand_volatility,
            inventory_level,
            order_quantity,
            risk_score,
            is_delayed
        ])

    columns = [
        "order_id",
        "supplier_id",
        "supplier_rating",
        "distance_km",
        "shipment_mode",
        "weather_score",
        "demand_volatility",
        "inventory_level",
        "order_quantity",
        "risk_score",
        "is_delayed"
    ]

    return pd.DataFrame(shipment_data, columns=columns)


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":
    suppliers = generate_suppliers()
    shipments = generate_shipments(suppliers)

    shipments.to_csv("data/supply_chain_shipments.csv", index=False)

    print("Dataset Generated Successfully!")
    print("Total Rows:", len(shipments))
    print("\nDelayed Distribution:")
    print(shipments["is_delayed"].value_counts(normalize=True))
