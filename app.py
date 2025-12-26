# ============================================
# app.py — FINAL COMBINED (Dark Neon, 15k/3k)
# ============================================

import os
import json
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tkinter as tk  # for Text widgets in popups
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk

# Metrics for 15k/3k evaluation
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# USGS-style helpers
from usgs_models import (
    classify_zone,
    etas_forecast,
    nowcast_score,
)

# =========================
# CONFIG
# =========================
FEATURES_CSV = "earthquake_featured.csv"

# 🔹 New model filenames for 15k/3k training/testing
REGRESSOR_FILE = "regressor_15k.pkl"
CLASSIFIER_FILE = "classifier_15k.pkl"

# Feature order JSON (same name as in train_all)
FEATURE_ORDER_FILE = "regressor_feature_order.json"

# =========================
# LOAD MODELS
# =========================
regressor = None
classifier = None
FEATURE_ORDER = None

if os.path.exists(REGRESSOR_FILE):
    try:
        regressor = joblib.load(REGRESSOR_FILE)
        print("Regressor (15k) loaded.")
    except Exception as e:
        print("Warning loading regressor:", e)
else:
    print(f"Regressor file not found: {REGRESSOR_FILE}")

if os.path.exists(CLASSIFIER_FILE):
    try:
        classifier = joblib.load(CLASSIFIER_FILE)
        print("Classifier (15k) loaded.")
    except Exception as e:
        print("Warning loading classifier:", e)
else:
    print(f"Classifier file not found: {CLASSIFIER_FILE}")

if os.path.exists(FEATURE_ORDER_FILE):
    try:
        with open(FEATURE_ORDER_FILE, "r", encoding="utf-8") as f:
            FEATURE_ORDER = json.load(f)
        print("Feature order loaded.")
    except Exception as e:
        print("Warning loading feature order JSON:", e)
else:
    print("Feature order JSON not found.")

# =========================
# LOAD CATALOG
# =========================
catalog = None
if os.path.exists(FEATURES_CSV):
    try:
        catalog = pd.read_csv(FEATURES_CSV)
        if "datetime" in catalog.columns:
            catalog["datetime"] = pd.to_datetime(catalog["datetime"], errors="coerce")
        catalog = catalog.sort_values("datetime").reset_index(drop=True)
        print("Catalog loaded:", FEATURES_CSV, "rows:", len(catalog))
    except Exception as e:
        print("Warning loading catalog:", e)
else:
    print(f"WARNING: {FEATURES_CSV} not found.")

# =========================
# HELPER FUNCTIONS
# =========================
def get_feature_order_or_infer(df: pd.DataFrame):
    """Return feature order from JSON or infer from catalog."""
    global FEATURE_ORDER
    if FEATURE_ORDER and isinstance(FEATURE_ORDER, list) and len(FEATURE_ORDER) > 0:
        return FEATURE_ORDER
    if df is None:
        return []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fallback = [c for c in numeric_cols if c not in ("magnitude", "sev_label", "eventid")]
    return fallback


def build_feature_row(**kwargs) -> pd.DataFrame:
    """
    Build a single engineered feature row for the ML models.
    Accepts keys like "Latitude", "longitude", etc.
    """
    def get_val(k, default=0):
        if k in kwargs:
            return kwargs[k]
        if k.lower() in kwargs:
            return kwargs[k.lower()]
        return default

    try:
        latitude = float(get_val("Latitude", get_val("latitude", 0.0)))
        longitude = float(get_val("Longitude", get_val("longitude", 0.0)))
        depth_km = float(get_val("Depth (km)", get_val("depth_km", 10.0)))
        nst = float(get_val("NST", get_val("nst", 0.0)))
        gap = float(get_val("Gap", get_val("gap", 0.0)))
        close = float(get_val("Close", get_val("close", 0.0)))
        rms = float(get_val("RMS", get_val("rms", 0.0)))
        year = int(get_val("Year", get_val("year", 2024)))
        month = int(get_val("Month", get_val("month", 1)))
        day = int(get_val("Day", get_val("day", 1)))
        hour = int(get_val("Hour", get_val("hour", 0)))
    except Exception as e:
        raise ValueError(f"Invalid input types: {e}")

    row = {
        "latitude": latitude,
        "longitude": longitude,
        "depth_km": depth_km,
        "nst": nst,
        "gap": gap,
        "close": close,
        "rms": rms,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
    }

    # cyclic encodings
    row["sin_month"] = np.sin(2 * np.pi * row["month"] / 12)
    row["cos_month"] = np.cos(2 * np.pi * row["month"] / 12)
    row["sin_hour"] = np.sin(2 * np.pi * row["hour"] / 24)
    row["cos_hour"] = np.cos(2 * np.pi * row["hour"] / 24)

    # history defaults (zero if not computed from catalog)
    hist_defaults = {
        "events_1d": 0, "events_7d": 0, "events_30d": 0,
        "max_mag_1d": 0.0, "max_mag_7d": 0.0, "max_mag_30d": 0.0,
        "mean_mag_7d": 0.0, "mean_mag_30d": 0.0,
        "energy_30d": 0.0, "b_value_30d": 0.0, "omori_p": 1.0
    }
    row.update(hist_defaults)

    return pd.DataFrame([row])


# =========================
# 15k / 3k HOLD-OUT METRICS
# =========================
HOLDOUT_METRICS = None

def compute_holdout_metrics():
    """
    Use first 15,000 rows as 'train' and next 3,000 as 'test' to
    evaluate the already-trained 15k models on a 3k hold-out set.
    """
    global HOLDOUT_METRICS

    if catalog is None or regressor is None or classifier is None:
        return

    df = catalog.copy()

    # require both targets
    if "magnitude" not in df.columns or "sev_label" not in df.columns:
        return

    df = df.dropna(subset=["magnitude", "sev_label"]).reset_index(drop=True)
    if len(df) < 18000:  # need at least 15k + 3k
        return

    n_train = 15000
    n_test = 3000

    feat_order = get_feature_order_or_infer(df)
    if not feat_order:
        return

    X_all = df[feat_order].select_dtypes(include=[np.number]).copy()
    y_reg_all = df["magnitude"].values
    y_clf_all = df["sev_label"].values

    X_test = X_all.iloc[n_train:n_train + n_test]
    y_reg_test = y_reg_all[n_train:n_train + n_test]
    y_clf_test = y_clf_all[n_train:n_train + n_test]

    # Regression metrics
    pred_reg = regressor.predict(X_test)
    r2 = r2_score(y_reg_test, pred_reg)
    mae = mean_absolute_error(y_reg_test, pred_reg)
    mse = mean_squared_error(y_reg_test, pred_reg)
    rmse = float(np.sqrt(mse))

    # Classification metrics
    pred_clf = classifier.predict(X_test)
    acc = accuracy_score(y_clf_test, pred_clf)
    prec = precision_score(y_clf_test, pred_clf, zero_division=0)
    rec = recall_score(y_clf_test, pred_clf, zero_division=0)
    f1 = f1_score(y_clf_test, pred_clf, zero_division=0)

    HOLDOUT_METRICS = {
        "r2": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
        "acc": float(acc),
        "prec": float(prec),
        "rec": float(rec),
        "f1": float(f1),
        "n_test": int(n_test),
        "n_train": int(n_train),
    }

compute_holdout_metrics()

# =========================
# PLOT GENERATION (fallback if PNGs missing)
# =========================
def generate_regression_plot():
    if regressor is None:
        raise RuntimeError("Regressor not loaded.")
    if catalog is None:
        raise RuntimeError("Catalog not loaded.")

    order = get_feature_order_or_infer(catalog)
    if not order:
        raise RuntimeError("No features available.")

    X = catalog[order].select_dtypes(include=[np.number]).copy()
    y = catalog["magnitude"].values

    preds = regressor.predict(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(y, preds, alpha=0.4)
    mn = min(np.nanmin(y), np.nanmin(preds))
    mx = max(np.nanmax(y), np.nanmax(preds))
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5)
    plt.xlabel("Actual Magnitude")
    plt.ylabel("Predicted Magnitude")
    plt.title("Regression: Actual vs Predicted Magnitude")
    plt.grid(True)
    plt.tight_layout()
    out = "regression_plot.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def generate_classification_plots():
    if classifier is None:
        raise RuntimeError("Classifier not loaded.")
    if catalog is None:
        raise RuntimeError("Catalog not loaded.")

    order = get_feature_order_or_infer(catalog)
    if not order:
        raise RuntimeError("No features available.")

    X = catalog[order].select_dtypes(include=[np.number]).copy()
    y_true = catalog["sev_label"].values
    y_pred = classifier.predict(X)

    from sklearn.metrics import (
        confusion_matrix,
        precision_score,
        recall_score,
        f1_score,
        accuracy_score,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Classification Confusion Matrix")
    plt.colorbar()
    classes = sorted(np.unique(np.concatenate((y_true, y_pred))))
    tick = np.arange(len(classes))
    plt.xticks(tick, classes)
    plt.yticks(tick, classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    plt.tight_layout()
    path_cm = "classification_cm.png"
    plt.savefig(path_cm, dpi=180)
    plt.close()

    # Metrics bar chart
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}
    plt.figure(figsize=(6, 4))
    names = list(metrics.keys())
    vals = [metrics[k] for k in names]
    bars = plt.bar(names, vals)
    plt.ylim(0, 1)
    plt.title("Classification Performance Metrics")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    path_metrics = "classification_metrics.png"
    plt.savefig(path_metrics, dpi=180)
    plt.close()

    return path_cm, path_metrics


# =========================
# POPUPS
# =========================
def show_image_popup(img_file, title=None):
    win = ttk.Toplevel(app)
    win.title(title or os.path.basename(img_file))
    win.geometry("920x720")

    canvas = ttk.Canvas(win)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar = ttk.Scrollbar(win, command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)

    frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    def update_scroll(event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    frame.bind("<Configure>", update_scroll)

    try:
        img = Image.open(img_file)
        max_width = 860
        if img.width > max_width:
            scale = max_width / img.width
            img = img.resize((int(img.width * scale), int(img.height * scale)))
        win.img_ref = ImageTk.PhotoImage(img)
        ttk.Label(frame, image=win.img_ref).pack(pady=10)
    except Exception as e:
        ttk.Label(
            frame,
            text=f"Error loading image {img_file}:\n{e}",
            bootstyle="danger"
        ).pack(padx=10, pady=20)


def show_text_popup(title, text):
    win = ttk.Toplevel(app)
    win.title(title)
    win.geometry("900x700")

    canvas = ttk.Canvas(win)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar = ttk.Scrollbar(win, command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.configure(yscrollcommand=scrollbar.set)

    frame = ttk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    def update_scroll(event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))

    frame.bind("<Configure>", update_scroll)

    ttk.Label(frame, text=title, font=("Segoe UI", 16, "bold")).pack(pady=10)

    txt = tk.Text(frame, width=100, height=35, wrap="word")
    txt.pack(padx=12, pady=6)
    txt.insert("1.0", text)
    txt.configure(state="disabled")


# =========================
# EXPLANATION TEXTS
# =========================
def explanation_for_prediction(row_df, mag_pred, sev_pred):
    lat = float(row_df["latitude"].iloc[0])
    lon = float(row_df["longitude"].iloc[0])
    depth = float(row_df["depth_km"].iloc[0])
    m = float(mag_pred)
    sev = int(sev_pred)

    def mmi_est(mag):
        if mag < 3:
            return "II–III (very weak)"
        if mag < 4:
            return "IV–V (light)"
        if mag < 5:
            return "VI–VII (moderate to strong)"
        if mag < 6:
            return "VII–VIII (strong to damaging)"
        return "VIII+ (very strong to severe)"

    mmi = mmi_est(m)

    explanation = f"""ML MAGNITUDE PREDICTION — Detailed Explanation

Input summary:
- Location: latitude={lat:.4f}, longitude={lon:.4f}
- Depth: {depth:.1f} km
- Features provided: NST, Gap, Close, RMS, and time-of-event (Y/M/D/H)

Model output:
- Predicted magnitude (ML regression): {m:.3f}
- Severity flag (classification): {sev}  ({'Severe — likely to cause noticeable damage' if sev==1 else 'Not Severe — unlikely to cause significant damage'})

Physical interpretation:
- Estimated Modified Mercalli Intensity (MMI): {mmi}
- Depth influences felt intensity: shallow earthquakes generally produce stronger shaking near the epicenter.

Uncertainty & caveats:
- This ML prediction is data-driven and relies on engineered features and historical examples.
- The regression predicts expected magnitude; real magnitudes can differ — treat this as an estimate, not a guarantee.

Guidance:
- If predicted magnitude ≥ 4 and severity flagged, consider increased alertness and preparedness.
- Combine this with Nowcast/ETAS outputs for short-term risk assessment.
"""
    return explanation


def explanation_for_zone(res):
    zone = res.get("zone")
    score = res.get("score")
    reason = res.get("reason", "")
    events_30 = res.get("events_30", 0)
    events_365 = res.get("events_365", 0)
    max30 = res.get("max30", 0.0)
    max365 = res.get("max365", 0.0)
    b = res.get("b_value", None)

    explanation = f"""ZONE CLASSIFICATION — Detailed Explanation

Assigned Zone: {zone} (1 = Very Low hazard, 5 = Very High hazard)
Score (internal): {score:.3f}

Why this zone was assigned:
- Recent activity: {events_30} events in last 30 days, {events_365} events in last 365 days.
- Maximum magnitudes: max30 = {max30:.2f}, max365 = {max365:.2f}.
- b-value indicator: {b if b is not None else 'Not enough data'}.
- Reason summary: {reason}

Zone meaning:
- Zone 1 — Very Low: very rare and weak seismic activity.
- Zone 2 — Low: low seismicity, minor damage from occasional moderate tremors.
- Zone 3 — Moderate: occasional damaging earthquakes possible.
- Zone 4 — High: frequent strong earthquakes with significant damage potential.
- Zone 5 — Very High: near major faults/plate boundaries; high risk of major, devastating earthquakes.

Guidance:
- Use Zone + Nowcast + ETAS together: a high Zone plus elevated short-term indicators suggests more urgent attention.
"""
    return explanation


def explanation_for_nowcast(res):
    score = res.get("score", 0.0)
    counts = res.get("counts", {})
    c1 = counts.get("1d", 0)
    c7 = counts.get("7d", 0)
    c30 = counts.get("30d", 0)
    c365 = counts.get("365d", 0)

    explanation = f"""USGS-STYLE NOWCAST — Detailed Explanation

Nowcast score (0–1): {score:.3f}

Recent event counts in your area:
- 1 day :  {c1}
- 7 days:  {c7}
- 30 days: {c30}
- 365 days: {c365}

Interpretation:
- Higher scores mean current seismicity is more active than usual in the chosen radius.
- Low scores (<0.2) suggest quiet periods; high scores (>0.6) indicate elevated activity.

Use:
- A high Nowcast score suggests recent clustering or aftershock sequences.
- Combine with ETAS (for triggered vs background events) and Zone for full context.
"""
    return explanation


def explanation_for_etas(res):
    tot = res.get("expected_total", 0.0)
    trig = res.get("expected_triggered", 0.0)
    bg = res.get("expected_background", 0.0)
    params = res.get("params", {})

    explanation = f"""ETAS SHORT-TERM FORECAST — Detailed Explanation

Forecast window: 1 day
Expected total events: {tot:.3f}
- Triggered (aftershock-type): {trig:.3f}
- Background (random/Poisson-like): {bg:.3f}

Model idea:
- ETAS = Epidemic-Type Aftershock Sequence model.
- Separates background seismicity (mu) from triggered events (K, alpha, p, etc.).
- Parameters used: {params}

Interpretation:
- When triggered component > background, aftershocks dominate.
- When background ≈ total, seismicity is back to normal levels.

Use:
- Helps understand whether recent mainshocks are likely to continue producing aftershocks.
- Supports decision-making around inspections, temporary closures, and alert levels.
"""
    return explanation


# =========================
# MAIN UI — DARK NEON (NO MAIN SCROLLBARS)
# =========================
app = ttk.Window(
    "Earthquake Dashboard — ML + USGS (Dark Neon, 15k/3k)",
    themename="cyborg"
)
app.geometry("1450x900")

ttk.Label(
    app,
    text="🌍 Earthquake Prediction • USGS-style Nowcasting • Hazard Zones",
    font=("Segoe UI", 20, "bold"),
    bootstyle="info"
).pack(pady=8)

main = ttk.Frame(app)
main.pack(fill="both", expand=True, padx=10, pady=10)

# LEFT: INPUT PANEL
left_frame = ttk.Frame(main)
left_frame.pack(side="left", fill="y", padx=(0, 20))

card_inputs = ttk.Frame(left_frame, padding=12, bootstyle="dark")
card_inputs.pack(fill="y")

ttk.Label(
    card_inputs,
    text="🧾 Earthquake Event Input",
    font=("Segoe UI", 14, "bold"),
    bootstyle="light"
).pack(pady=6, anchor="w")

ttk.Label(
    card_inputs,
    text="Enter hypothetical event location, depth and origin time.\n"
         "These values are used by the ML models and USGS-style analysis.",
    wraplength=350,
    justify="left"
).pack(pady=(0, 8), anchor="w")

fields = ["Latitude", "Longitude", "Depth (km)", "NST", "Gap", "Close", "RMS", "Year", "Month", "Day", "Hour"]
defaults = [28.6, 77.2, 10, 10, 100, 5, 0.1, 2024, 1, 1, 12]
entry = {}

frm_inputs = ttk.Frame(card_inputs)
frm_inputs.pack()

for i, (f, d) in enumerate(zip(fields, defaults)):
    ttk.Label(frm_inputs, text=f).grid(row=i, column=0, sticky="w", pady=3)
    e = ttk.Entry(frm_inputs, width=14)
    e.insert(0, str(d))
    e.grid(row=i, column=1, pady=3, padx=3)
    entry[f] = e

ttk.Label(
    card_inputs,
    text="Tip: Latitude/Longitude in degrees, Depth in km,\n"
         "NST/Gap/Close/RMS are station/quality parameters.",
    wraplength=350,
    justify="left",
    bootstyle="secondary"
).pack(pady=8, anchor="w")

# small inline summaries left side
ml_res_lbl = ttk.Label(
    card_inputs,
    text="ML Output: (Press 'Predict Magnitude & Severity')",
    font=("Segoe UI", 10),
    anchor="w",
    justify="left",
    wraplength=350,
)
ml_res_lbl.pack(pady=4, anchor="w")

zone_result = ttk.Label(
    card_inputs,
    text="Zone: (Press 'Classify Seismic Hazard Zone')",
    font=("Segoe UI", 10),
    anchor="w",
    justify="left",
    wraplength=350,
)
zone_result.pack(pady=3, anchor="w")

nowcast_label = ttk.Label(
    card_inputs,
    text="Nowcast: (Press 'Compute USGS-Style Nowcast')",
    font=("Segoe UI", 9),
    anchor="w",
    justify="left",
    wraplength=350,
)
nowcast_label.pack(pady=3, anchor="w")

etas_label = ttk.Label(
    card_inputs,
    text="ETAS: (Press 'Run 1-Day ETAS Aftershock Forecast')",
    font=("Segoe UI", 9),
    anchor="w",
    justify="left",
    wraplength=350,
)
etas_label.pack(pady=3, anchor="w")

# globals for right side status summaries
last_pred_summary = tk.StringVar(value="No predictions yet.")
last_zone_summary = tk.StringVar(value="No zone classified yet.")
last_nowcast_summary = tk.StringVar(value="No nowcast run yet.")
last_etas_summary = tk.StringVar(value="No ETAS forecast run yet.")

# RIGHT: ACTION + VISUALIZATION PANEL
right_frame = ttk.Frame(main)
right_frame.pack(side="right", fill="both", expand=True)

# ---- ACTION CENTER (buttons on right, between input & analysis) ----
action_card = ttk.Frame(right_frame, padding=15, bootstyle="dark")
action_card.pack(fill="x", pady=(0, 10))

ttk.Label(
    action_card,
    text="⚡ Analysis Controls",
    font=("Segoe UI", 15, "bold"),
    bootstyle="light"
).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

ttk.Label(
    action_card,
    text=(
        "Buttons explained:\n"
        "• Predict Magnitude & Severity (ML Model): Use the machine learning models to estimate\n"
        "  earthquake magnitude and whether it is 'severe' (≥4) or 'non-severe'.\n"
        "• Classify Seismic Hazard Zone (1–5): Assigns the location to a zone from Very Low (1)\n"
        "  to Very High hazard (5), based on local seismicity.\n"
        "• Compute USGS-Style Nowcast (1 Day): Measures how active the region is recently using\n"
        "  event counts in multiple time windows.\n"
        "• Run 1-Day ETAS Aftershock Forecast: Uses an ETAS model to estimate triggered vs\n"
        "  background events expected in the next day."
    ),
    wraplength=800,
    justify="left"
).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

# Button callbacks
def on_predict_button():
    try:
        vals = {
            f: (float(entry[f].get()) if f not in ("Year", "Month", "Day", "Hour") else int(entry[f].get()))
            for f in fields
        }

        row = build_feature_row(**vals)

        feat_order = get_feature_order_or_infer(catalog)
        if feat_order:
            for c in feat_order:
                if c not in row.columns:
                    row[c] = 0.0
            row = row[feat_order]
        else:
            row = row.select_dtypes(include=[np.number])

        if regressor is None or classifier is None:
            raise RuntimeError("Regressor or Classifier not loaded.")

        mag = float(regressor.predict(row)[0])
        sev = int(classifier.predict(row)[0])

        sev_str = "Severe (≥4.0)" if sev == 1 else "Not Severe (<4.0)"
        ml_res_lbl.config(text=f"ML Output → Magnitude: {mag:.3f} | Severity: {sev_str}")
        last_pred_summary.set(
            f"Predicted magnitude: {mag:.3f}\n"
            f"Predicted severity: {sev_str}\n"
            f"Location: ({vals['Latitude']:.3f}, {vals['Longitude']:.3f})\n"
            f"Origin time: {vals['Year']}-{vals['Month']:02d}-{vals['Day']:02d} {vals['Hour']:02d}:00"
        )

        expl = explanation_for_prediction(build_feature_row(**vals), mag, sev)
        show_text_popup("🔍 ML Prediction Explanation", expl)

    except Exception as e:
        ml_res_lbl.config(text=f"Error: {e}")
        show_text_popup("ML Prediction Error", f"An error occurred while predicting:\n\n{e}")


def on_zone_button():
    try:
        lat = float(entry["Latitude"].get())
        lon = float(entry["Longitude"].get())

        if catalog is None:
            raise RuntimeError("Catalog not loaded — cannot classify zone.")

        res = classify_zone(lat, lon, catalog)
        zone_result.config(text=f"Zone: {res['zone']} | Score: {res['score']:.2f}")
        last_zone_summary.set(
            f"Zone: {res['zone']} (1=Very Low, 5=Very High)\n"
            f"Score: {res['score']:.2f}\nReason: {res['reason']}"
        )

        expl = explanation_for_zone(res)
        show_text_popup("🗺 Zone Classification Explanation", expl)

    except Exception as e:
        zone_result.config(text=f"Zone error: {e}")
        show_text_popup("Zone Classification Error", f"An error occurred:\n\n{e}")


def on_nowcast_button():
    try:
        lat = float(entry["Latitude"].get())
        lon = float(entry["Longitude"].get())

        if catalog is None:
            raise RuntimeError("Catalog not loaded — cannot run nowcast.")

        origin = catalog["datetime"].max()
        res = nowcast_score(catalog, origin, lat, lon)

        nowcast_label.config(
            text=f"Nowcast score: {res['score']:.3f} | Counts: {res.get('counts',{})}"
        )
        last_nowcast_summary.set(
            f"Nowcast score: {res['score']:.3f}\nCounts: {res.get('counts',{})}"
        )

        expl = explanation_for_nowcast(res)
        show_text_popup("📈 Nowcast Explanation", expl)

    except Exception as e:
        nowcast_label.config(text=f"Nowcast error: {e}")
        show_text_popup("Nowcast Error", f"An error occurred:\n\n{e}")


def on_etas_button():
    try:
        if catalog is None:
            raise RuntimeError("Catalog not loaded — cannot run ETAS.")
        origin = catalog["datetime"].max()
        res = etas_forecast(catalog, origin, window_days=1.0)

        etas_label.config(
            text=f"ETAS expected total: {res['expected_total']:.3f} "
                 f"(Triggered: {res['expected_triggered']:.3f}, "
                 f"Background: {res['expected_background']:.3f})"
        )
        last_etas_summary.set(
            f"Expected total (1 day): {res['expected_total']:.3f}\n"
            f"Triggered: {res['expected_triggered']:.3f}\n"
            f"Background: {res['expected_background']:.3f}"
        )

        expl = explanation_for_etas(res)
        show_text_popup("🚨 ETAS Forecast Explanation", expl)

    except Exception as e:
        etas_label.config(text=f"ETAS error: {e}")
        show_text_popup("ETAS Error", f"An error occurred:\n\n{e}")


# Buttons placed in 2x2 grid in the action center
ttk.Button(
    action_card,
    text="🔍 Predict Magnitude & Severity (ML Model)",
    bootstyle=SUCCESS,
    command=on_predict_button
).grid(row=2, column=0, padx=5, pady=5, sticky="ew")

ttk.Button(
    action_card,
    text="🗺 Classify Seismic Hazard Zone (1–5)",
    bootstyle=PRIMARY,
    command=on_zone_button
).grid(row=2, column=1, padx=5, pady=5, sticky="ew")

ttk.Button(
    action_card,
    text="📈 Compute USGS-Style Nowcast (1 Day)",
    bootstyle=WARNING,
    command=on_nowcast_button
).grid(row=3, column=0, padx=5, pady=5, sticky="ew")

ttk.Button(
    action_card,
    text="🚨 Run 1-Day ETAS Aftershock Forecast",
    bootstyle=INFO,
    command=on_etas_button
).grid(row=3, column=1, padx=5, pady=5, sticky="ew")

action_card.columnconfigure(0, weight=1)
action_card.columnconfigure(1, weight=1)

# ---- VISUALIZATION & EVALUATION SECTION ----
card_right = ttk.Frame(right_frame, padding=12, bootstyle="dark")
card_right.pack(fill="both", expand=True)

ttk.Label(
    card_right,
    text="📊 Analysis & Visualizations",
    font=("Segoe UI", 14, "bold"),
    bootstyle="light"
).pack(pady=4, anchor="w")

notebook = ttk.Notebook(card_right, bootstyle="dark")
notebook.pack(fill="both", expand=True, pady=6)

tab_overview = ttk.Frame(notebook, padding=10)
tab_pred = ttk.Frame(notebook, padding=10)
tab_usgs = ttk.Frame(notebook, padding=10)
tab_plots = ttk.Frame(notebook, padding=10)

notebook.add(tab_overview, text="Overview")
notebook.add(tab_pred, text="Prediction Details")
notebook.add(tab_usgs, text="USGS Models")
notebook.add(tab_plots, text="Plots")

# ---- Overview tab with a dedicated scrollbar ----

ov_canvas = ttk.Canvas(tab_overview)
ov_canvas.pack(side="left", fill="both", expand=True)

ov_scrollbar = ttk.Scrollbar(tab_overview, orient="vertical", command=ov_canvas.yview)
ov_scrollbar.pack(side="right", fill="y")
ov_canvas.configure(yscrollcommand=ov_scrollbar.set)

ov_frame = ttk.Frame(ov_canvas, padding=10)
ov_canvas.create_window((0, 0), window=ov_frame, anchor="nw")

def _update_ov_scroll(event=None):
    ov_canvas.configure(scrollregion=ov_canvas.bbox("all"))

ov_frame.bind("<Configure>", _update_ov_scroll)

ttk.Label(
    ov_frame,
    text="🌐 Dashboard Overview",
    font=("Segoe UI", 13, "bold")
).pack(pady=6, anchor="w")

overview_text = (
    f"• Engineered catalog: {FEATURES_CSV}\n"
    f"• Rows: {len(catalog) if catalog is not None else 'N/A'}\n"
    f"• Regressor (15k): {'Loaded' if regressor is not None else 'Missing'}\n"
    f"• Classifier (15k): {'Loaded' if classifier is not None else 'Missing'}\n\n"
    "Use the inputs on the left and the analysis controls above to:\n"
    "  - Predict earthquake magnitude & severity using ML.\n"
    "  - Classify seismic hazard zone (1–5).\n"
    "  - Compute a USGS-style Nowcast score for recent activity.\n"
    "  - Run an ETAS short-term aftershock forecast.\n"
)
ttk.Label(ov_frame, text=overview_text, justify="left", anchor="w", wraplength=800).pack(pady=4, anchor="w")

if HOLDOUT_METRICS is not None:
    hm = HOLDOUT_METRICS
    eval_text = (
        f"📌 Model Performance (15k train / 3k test)\n"
        f"   — Regression (Magnitude):\n"
        f"      • R² Score : {hm['r2']:.4f}\n"
        f"      • MAE      : {hm['mae']:.4f}\n"
        f"      • RMSE     : {hm['rmse']:.4f}\n\n"
        f"   — Classification (Severity ≥ 4):\n"
        f"      • Accuracy : {hm['acc']:.4f}\n"
        f"      • Precision: {hm['prec']:.4f}\n"
        f"      • Recall   : {hm['rec']:.4f}\n"
        f"      • F1 Score : {hm['f1']:.4f}\n"
        f"   (Test set size: {hm['n_test']} samples)\n"
    )
else:
    eval_text = (
        "📌 Model Performance (15k train / 3k test):\n"
        "   Not available — need ≥ 18k rows.\n"
    )

ttk.Label(ov_frame, text=eval_text, justify="left", anchor="w", wraplength=800, bootstyle="info").pack(pady=8, anchor="w")

ttk.Label(ov_frame, text="Last ML Prediction Summary:", font=("Segoe UI", 11, "bold")).pack(pady=(10, 2), anchor="w")
ttk.Label(ov_frame, textvariable=last_pred_summary, justify="left", wraplength=800).pack(anchor="w")

ttk.Label(ov_frame, text="Last Zone Classification:", font=("Segoe UI", 11, "bold")).pack(pady=(10, 2), anchor="w")
ttk.Label(ov_frame, textvariable=last_zone_summary, justify="left", wraplength=800).pack(anchor="w")

# ---- Prediction details tab ----
ttk.Label(
    tab_pred,
    text="🔍 Prediction Details",
    font=("Segoe UI", 13, "bold")
).pack(pady=6, anchor="w")

ttk.Label(
    tab_pred,
    text="This tab mirrors the latest ML prediction and zone output.\n"
         "For a full technical explanation, read the popups that appear when you press each button.",
    wraplength=800,
    justify="left"
).pack(pady=4, anchor="w")

ttk.Label(
    tab_pred,
    text="Latest ML Prediction:",
    font=("Segoe UI", 11, "bold")
).pack(pady=(10, 2), anchor="w")
ttk.Label(tab_pred, textvariable=last_pred_summary, justify="left", wraplength=800).pack(anchor="w")

ttk.Label(
    tab_pred,
    text="Latest Zone Assessment:",
    font=("Segoe UI", 11, "bold")
).pack(pady=(10, 2), anchor="w")
ttk.Label(tab_pred, textvariable=last_zone_summary, justify="left", wraplength=800).pack(anchor="w")

# ---- USGS models tab ----
ttk.Label(
    tab_usgs,
    text="📡 USGS-Style Models (Nowcast & ETAS)",
    font=("Segoe UI", 13, "bold")
).pack(pady=6, anchor="w")

ttk.Label(
    tab_usgs,
    text="Short-term activity indicators based on your catalog:",
    wraplength=800,
    justify="left"
).pack(pady=4, anchor="w")

ttk.Label(
    tab_usgs,
    text="Latest Nowcast:",
    font=("Segoe UI", 11, "bold")
).pack(pady=(10, 2), anchor="w")
ttk.Label(tab_usgs, textvariable=last_nowcast_summary, justify="left", wraplength=800).pack(anchor="w")

ttk.Label(
    tab_usgs,
    text="Latest ETAS Forecast:",
    font=("Segoe UI", 11, "bold")
).pack(pady=(10, 2), anchor="w")
ttk.Label(tab_usgs, textvariable=last_etas_summary, justify="left", wraplength=800).pack(anchor="w")

# ---- Plots tab ----
ttk.Label(
    tab_plots,
    text="📈 Model Plots",
    font=("Segoe UI", 13, "bold")
).pack(pady=6, anchor="w")

ttk.Label(
    tab_plots,
    text=(
        "You can view:\n"
        "  • Regression: Actual vs Predicted Magnitude (saved PNG)\n"
        "  • Classification: Confusion Matrix (saved PNG)\n\n"
        "If the PNG files are missing, the app will recompute the plots from the models."
    ),
    wraplength=800,
    justify="left"
).pack(pady=4, anchor="w")

def on_show_regression():
    try:
        img_file = "regression_plot.png"
        if not os.path.exists(img_file):
            # Fallback to computing from models if PNG missing
            img_file = generate_regression_plot()
        show_image_popup(img_file, title="Regression: Actual vs Predicted Magnitude")
    except Exception as e:
        show_text_popup("Regression Plot Error", f"Error showing regression plot:\n\n{e}")


def on_show_classification():
    try:
        img_file = "classification_plot.png"
        if os.path.exists(img_file):
            show_image_popup(img_file, title="Classification — Confusion Matrix")
        else:
            # Fallback to computing confusion matrix + metrics if PNG missing
            cm_path, metrics_path = generate_classification_plots()
            show_image_popup(cm_path, title="Classification — Confusion Matrix")
            show_image_popup(metrics_path, title="Classification — Metrics")
    except Exception as e:
        show_text_popup("Classification Plots Error", f"Error showing classification plots:\n\n{e}")

ttk.Button(
    tab_plots,
    text="📊 Show Regression Plot",
    bootstyle=PRIMARY,
    command=on_show_regression
).pack(pady=8, anchor="w")

ttk.Button(
    tab_plots,
    text="📊 Show Classification Plot",
    bootstyle=INFO,
    command=on_show_classification
).pack(pady=4, anchor="w")

# ---- Status bar ----
status_txt = (
    f"Catalog: {FEATURES_CSV if os.path.exists(FEATURES_CSV) else 'MISSING'}  |  "
    f"Regressor (15k): {'Loaded' if regressor is not None else 'Missing'}  |  "
    f"Classifier (15k): {'Loaded' if classifier is not None else 'Missing'}"
)
ttk.Label(
    card_right,
    text=status_txt,
    font=("Segoe UI", 9),
    bootstyle="secondary"
).pack(pady=4, anchor="w")

# =========================
# RUN APP
# =========================
app.mainloop()
