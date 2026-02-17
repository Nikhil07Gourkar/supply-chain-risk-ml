import pandas as pd
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# ==========================
# 1. Load Dataset
# ==========================
df = pd.read_csv("data/supply_chain_shipments.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# ==========================
# 2. Encode Categorical Data
# ==========================
df = pd.get_dummies(df, columns=["shipment_mode"], drop_first=True)

# Drop ID columns
df = df.drop(columns=["order_id", "supplier_id"])

# ==========================
# 3. Define Features & Target
# ==========================
X = df.drop("is_delayed", axis=1)
y = df["is_delayed"]

# ==========================
# 4. Handle Class Imbalance
# ==========================
counter = Counter(y)
scale_pos_weight = counter[0] / counter[1]

print("Class Distribution:", counter)
print("Scale Pos Weight:", scale_pos_weight)

# ==========================
# 5. Train-Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================
# 6. Train XGBoost Model
# ==========================
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# ==========================
# 7. Predictions
# ==========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ==========================
# 8. Evaluation
# ==========================
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_prob)
print("\nROC AUC Score:", roc_auc)

# ==========================
# 9. Feature Importance Plot
# ==========================
importance = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importance)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()

# ==========================
# 10. Save Model
# ==========================
joblib.dump(model, "supply_chain_xgb_model.pkl")
print("\nModel saved as supply_chain_xgb_model.pkl")
