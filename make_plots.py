# ================================================================
# make_plots.py — Plot Train & Test Performance from train_all.py
# ================================================================
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ------------------------------
#Files from train_all.py
# ------------------------------
FEATURES_CSV = "earthquake_featured.csv"
TEST_CSV = "earthquake_test.csv"
REGRESSOR = "regressor.pkl"
CLASSIFIER = "classifier.pkl"
FEATURE_ORDER_JSON = "regressor_feature_order.json"

print("\nLoading files...")

# Load dataset
df = pd.read_csv(FEATURES_CSV)
df["datetime"] = pd.to_datetime(df["datetime"], errors="ignore")
df = df.dropna(subset=["magnitude", "sev_label"])

# Load test set
test_df = pd.read_csv(TEST_CSV)

# Load models
reg = joblib.load(REGRESSOR)
clf = joblib.load(CLASSIFIER)

# Load feature order
with open(FEATURE_ORDER_JSON, "r") as f:
    features = json.load(f)

print("✓ Loaded models, feature order & data")

# ------------------------------------------
# Separate Train and Test
# ------------------------------------------
train_df = df.drop(test_df.index)

X_train = train_df[features]
y_train_reg = train_df["magnitude"]
y_train_clf = train_df["sev_label"]

X_test = test_df[features]
y_test_reg = test_df["magnitude"]
y_test_clf = test_df["sev_label"]

print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}")

# ================================================================
# Helper: Regression Plot
# ================================================================
def plot_regression(y_true, y_pred, title, fname):
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel("Actual Magnitude")
    plt.ylabel("Predicted Magnitude")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=250)
    plt.close()
    print(f"✔ Saved {fname}")

# ================================================================
# Helper: Confusion Matrix Plot
# ================================================================
def plot_cm(y_true, y_pred, title, fname, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=cmap, values_format='d')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=250)
    plt.close()
    print(f"✔ Saved {fname}")

# ================================================================
# Generate Plots
# ================================================================
print("\nGenerating plots...")

# Regression
plot_regression(y_train_reg, reg.predict(X_train),
                "Regression Train Performance", "regression_train.png")

plot_regression(y_test_reg, reg.predict(X_test),
                "Regression Test Performance", "regression_test.png")

# Classification
plot_cm(y_train_clf, clf.predict(X_train),
        "Confusion Matrix - Train", "cm_train.png", cmap="Purples")

plot_cm(y_test_clf, clf.predict(X_test),
        "Confusion Matrix - Test", "cm_test.png", cmap="Greens")

print("\n🎉 All plots created successfully!")
