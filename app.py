import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load trained model
# -------------------------
model = joblib.load("supply_chain_xgb_model.pkl")

st.title("🚚 Supply Chain Delay Prediction System")

st.write("Enter shipment details to predict delay risk.")

# -------------------------
# User Inputs
# -------------------------

supplier_rating = st.slider("Supplier Rating", 1, 5, 3)
distance_km = st.number_input("Distance (km)", 50, 2000, 500)
weather_score = st.slider("Weather Risk Score", 0.0, 1.0, 0.3)
demand_volatility = st.slider("Demand Volatility", 0.0, 1.0, 0.4)
inventory_level = st.number_input("Inventory Level", 0, 1000, 500)
order_quantity = st.number_input("Order Quantity", 50, 500, 200)
shipment_mode = st.selectbox("Shipment Mode", ["Air", "Sea", "Road"])

# Risk score approximation (simple formula for dashboard consistency)
risk_score = int((weather_score + demand_volatility) * 50)

# -------------------------
# Convert Input to DataFrame
# -------------------------

input_data = {
    "supplier_rating": supplier_rating,
    "distance_km": distance_km,
    "weather_score": weather_score,
    "demand_volatility": demand_volatility,
    "inventory_level": inventory_level,
    "order_quantity": order_quantity,
    "risk_score": risk_score,
    "shipment_mode_Road": 1 if shipment_mode == "Road" else 0,
    "shipment_mode_Sea": 1 if shipment_mode == "Sea" else 0,
}

input_df = pd.DataFrame([input_data])

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Delay Risk"):

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    st.write(f"Delay Probability: {probability:.2%}")

    if prediction == 1:
        st.error("⚠ High Risk of Delay")
    else:
        st.success("✅ Low Risk of Delay")
