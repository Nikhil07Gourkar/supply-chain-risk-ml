# 🚚 Supply Chain Risk Prediction System

🔗 **Live App:** https://supply-chain-risk-ml-2jdhtnrncgxqovd8syfxmu.streamlit.app/  
📂 **GitHub Repo:** https://github.com/Nikhil07Gourkar/supply-chain-risk-ml  

---

## 📌 Project Overview

This project predicts whether a shipment will be delayed using Machine Learning.

It uses **XGBoost** to classify delivery risk based on supply chain features such as distance, weather conditions, inventory levels, supplier performance, and more.

The model is deployed as a **live interactive Streamlit dashboard** where users can input shipment details and instantly get delay predictions.

---

## 🚀 Live Demo

👉 Try it here:  
**https://supply-chain-risk-ml-2jdhtnrncgxqovd8syfxmu.streamlit.app/**

Users can:
- Enter shipment details
- Get real-time delay prediction
- View delay probability score

---

## 🧠 Machine Learning Details

- Model: **XGBoost Classifier**
- Dataset Size: **30,000 rows (synthetic supply chain data)**
- Features: 10 input variables
- Target: `is_delayed` (0 = On Time, 1 = Delayed)

### 📊 Model Performance

- Accuracy: **89%**
- ROC AUC Score: **0.96**
- Precision (Delayed Class): **0.92**
- Recall (Delayed Class): **0.88**

The model demonstrates strong predictive performance and good class balance handling.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Git & GitHub
- Streamlit Cloud (Deployment)

