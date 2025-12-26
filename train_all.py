# ==========================================
# train_all.py - Improved with Train/Test Split & Evaluation
# ==========================================

import json
import joblib
import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score
)

# ------------------------------------------
# CONFIG
# ------------------------------------------
FEATURES_CSV = "earthquake_featured.csv"
REGRESSOR_OUT = "regressor_15k.pkl"
CLASSIFIER_OUT = "classifier_15k.pkl"
FEATURE_ORDER_OUT = "regressor_feature_order.json"
TEST_OUT_CSV = "earthquake_test.csv"
EVAL_OUT_JSON = "model_eval.json"

# ------------------------------------------
# LOAD DATA
# ------------------------------------------
print("\nLoading:", FEATURES_CSV)
df = pd.read_csv(FEATURES_CSV)
df["datetime"] = pd.to_datetime(df["datetime"], errors="ignore")

# numeric feature columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

exclude = {"magnitude", "sev_label", "eventid"}
features = [c for c in numeric_cols if c not in exclude]

print(f"Training with {len(features)} engineered features")

X = df[features]
y_reg = df["magnitude"]
y_clf = df["sev_label"]

# ------------------------------------------
# TRAIN-TEST SPLIT (approx 15k train, 3k test)
# ------------------------------------------
X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
    X, y_reg, y_clf, test_size=0.17, random_state=42, shuffle=True
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

pd.concat([X_test, y_test_reg, y_test_clf], axis=1).to_csv(TEST_OUT_CSV, index=False)
print("Saved testing dataset ->", TEST_OUT_CSV)

# ------------------------------------------
# Train LightGBM Regressor
# ------------------------------------------
print("\nTraining LightGBM Regressor...")
regressor = LGBMRegressor(
    n_estimators=1200,
    learning_rate=0.015,
    subsample=0.85,
    colsample_bytree=0.8,
    random_state=42
)
regressor.fit(X_train, y_train_reg)

pred_reg = regressor.predict(X_test)
r2 = r2_score(y_test_reg, pred_reg)
mae = mean_absolute_error(y_test_reg, pred_reg)
rmse = np.sqrt(mean_squared_error(y_test_reg, pred_reg))

print("\n📌 Regression Evaluation:")
print(f"R2 Score  : {r2:.4f}")
print(f"MAE       : {mae:.4f}")
print(f"RMSE      : {rmse:.4f}")

# ------------------------------------------
# Train Random Forest Classifier
# ------------------------------------------
print("\nTraining RandomForest Classifier...")
clf = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    random_state=42
)
clf.fit(X_train, y_train_clf)

pred_clf = clf.predict(X_test)
acc = accuracy_score(y_test_clf, pred_clf)
prec = precision_score(y_test_clf, pred_clf, zero_division=0)
rec = recall_score(y_test_clf, pred_clf, zero_division=0)
f1 = f1_score(y_test_clf, pred_clf, zero_division=0)

print("\n📌 Classification Evaluation:")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")

# ------------------------------------------
# SAVE MODELS
# ------------------------------------------
joblib.dump(regressor, REGRESSOR_OUT)
joblib.dump(clf, CLASSIFIER_OUT)
print("\nSaved models ✔")

# ------------------------------------------
# SAVE FEATURE ORDER (required for app)
# ------------------------------------------
with open(FEATURE_ORDER_OUT, "w") as f:
    json.dump(features, f, indent=2)
print("Saved feature order ✔")

# ------------------------------------------
# SAVE EVALUATION RESULTS
# ------------------------------------------
eval_report = {
    "regression": {"R2": r2, "MAE": mae, "RMSE": rmse},
    "classification": {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
}
with open(EVAL_OUT_JSON, "w") as f:
    json.dump(eval_report, f, indent=2)

print("\nSaved evaluation report ->", EVAL_OUT_JSON)
print("\n🎯 Training Completed Successfully!")
