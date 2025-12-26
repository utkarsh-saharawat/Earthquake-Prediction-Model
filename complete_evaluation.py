# complete_evaluation.py
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score
)

from usgs_models import classify_zone, nowcast_score

FEATURES_CSV = "earthquake_featured.csv"
REGRESSOR = "regressor.pkl"
CLASSIFIER = "classifier.pkl"
FEATURE_ORDER_JSON = "regressor_feature_order.json"

print("Loading dataset...")
df = pd.read_csv(FEATURES_CSV)

# ---- numeric features ----
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude = {"magnitude", "sev_label", "eventid"}
features = [c for c in numeric_cols if c not in exclude]

print("Using features:", features)

# ---- load models ----
print("\nLoading regressor...")
reg = joblib.load(REGRESSOR)

print("Loading classifier...")
clf = joblib.load(CLASSIFIER)

# ============================================================
# REGRESSION EVALUATION
# ============================================================
print("\n=== REGRESSION METRICS ===")

X_reg = df[features]
y_reg = df["magnitude"]

pred_reg = reg.predict(X_reg)

r2 = r2_score(y_reg, pred_reg)
mae = mean_absolute_error(y_reg, pred_reg)
mse = mean_squared_error(y_reg, pred_reg)
rmse = np.sqrt(mse)

print(f"R²  : {r2:.4f}")
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# ============================================================
# CLASSIFIER EVALUATION
# ============================================================
print("\n=== CLASSIFICATION METRICS ===")

X_clf = df[features]
y_clf = df["sev_label"]

pred_clf = clf.predict(X_clf)
clf_acc = accuracy_score(y_clf, pred_clf)

print(f"Accuracy: {clf_acc:.4f}")

# ============================================================
# USGS ZONE MODEL EVALUATION
# ============================================================
print("\n=== USGS-STYLE ZONE MODEL ===")

# Ensure datetime
if not np.issubdtype(df["datetime"].dtype, np.datetime64):
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

zone_preds = []
for _, row in df.iterrows():
    out = classify_zone(
        row["latitude"],
        row["longitude"],
        df
    )
    zone_preds.append(out["zone"])

df["zone_pred"] = zone_preds

# Simple histogram
zone_dist = df["zone_pred"].value_counts().sort_index()

print("\nZone prediction distribution:")
for z, c in zone_dist.items():
    print(f"  Zone {z}: {c} cases")

# ============================================================
# NOWCAST EVALUATION (OPTIONAL)
# ============================================================
print("\n=== NOWCAST SCORE (USGS COUNT MODEL) ===")

origin = df["datetime"].max()
lat0 = df["latitude"].mean()
lon0 = df["longitude"].mean()

n_out = nowcast_score(df, origin, lat0, lon0)

print(f"Nowcast Score at dataset center: {n_out['score']:.4f}")
print("Counts:", n_out["counts"])

# ============================================================
# SUMMARY
# ============================================================
print("\n===== FINAL SUMMARY =====")
print(f"Regression R²      : {r2:.4f}")
print(f"Regression RMSE    : {rmse:.4f}")
print(f"Classifier Accuracy: {clf_acc:.4f}")
print("Zone model: predicted zones →", dict(zone_dist))

print("\nDONE.")
