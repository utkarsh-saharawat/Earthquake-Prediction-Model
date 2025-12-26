import pandas as pd
import numpy as np
from tqdm import tqdm

# ==========================================================
# FIXED OMORI FUNCTION — works with numpy.timedelta64
# ==========================================================
def omori_p(times):
    if len(times) < 2:
        return 1.0

    # Convert to pandas datetime (ensures compatibility)
    t = pd.to_datetime(times)

    # Differences between consecutive events
    diffs = np.diff(t)

    # Convert to seconds
    diffs_sec = diffs / np.timedelta64(1, "s")

    # Positive intervals only
    diffs_sec = diffs_sec[diffs_sec > 0]

    if len(diffs_sec) == 0:
        return 1.0

    # Omori-style decay estimation: p ≈ median(1/Δt)
    return float(np.median(1.0 / diffs_sec))


# ==========================================================
# LOAD DATA
# ==========================================================
df = pd.read_csv("earthquake_cleaned.csv")

# Ensure datetime exists
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)

# ==========================================================
# ADD NEW FEATURE COLUMNS
# ==========================================================
df["events_1d"] = 0
df["events_7d"] = 0
df["events_30d"] = 0

df["max_mag_1d"] = 0.0
df["max_mag_7d"] = 0.0
df["max_mag_30d"] = 0.0

df["mean_mag_7d"] = 0.0
df["mean_mag_30d"] = 0.0

df["energy_30d"] = 0.0
df["b_value_30d"] = 0.0

df["omori_p"] = 1.0

# Time-cyclic features
df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

# ==========================================================
# TIME WINDOWS
# ==========================================================
window_1d = pd.Timedelta(days=1)
window_7d = pd.Timedelta(days=7)
window_30d = pd.Timedelta(days=30)

print("Engineering time-history seismic features...")
for i in tqdm(range(len(df))):
    t = df.loc[i, "datetime"]

    # Filter windows
    win_1 = df[(df["datetime"] > t - window_1d) & (df["datetime"] < t)]
    win_7 = df[(df["datetime"] > t - window_7d) & (df["datetime"] < t)]
    win_30 = df[(df["datetime"] > t - window_30d) & (df["datetime"] < t)]

    # Event counts
    df.loc[i, "events_1d"] = len(win_1)
    df.loc[i, "events_7d"] = len(win_7)
    df.loc[i, "events_30d"] = len(win_30)

    # Max magnitudes
    df.loc[i, "max_mag_1d"] = win_1["magnitude"].max() if len(win_1) else 0
    df.loc[i, "max_mag_7d"] = win_7["magnitude"].max() if len(win_7) else 0
    df.loc[i, "max_mag_30d"] = win_30["magnitude"].max() if len(win_30) else 0

    # Mean magnitudes
    df.loc[i, "mean_mag_7d"] = win_7["magnitude"].mean() if len(win_7) else 0
    df.loc[i, "mean_mag_30d"] = win_30["magnitude"].mean() if len(win_30) else 0

    # Energy release (10^(1.5M))
    if len(win_30):
        df.loc[i, "energy_30d"] = win_30.apply(
            lambda r: 10 ** (1.5 * r["magnitude"]), axis=1
        ).sum()

    # b-value from Gutenberg-Richter law
    if len(win_30) >= 5:
        mags = win_30["magnitude"]
        mmin = mags.min()
        mean_mag = mags.mean()

        # Avoid division by zero
        if mean_mag > mmin:
            b = 1 / ((mean_mag - mmin) * np.log(10))
            df.loc[i, "b_value_30d"] = b

    # Omori p-value (fixed version)
    df.loc[i, "omori_p"] = omori_p(win_7["datetime"].values)


# ==========================================================
# SAVE OUTPUT
# ==========================================================
df.to_csv("earthquake_featured.csv", index=False)
print("\n🔥 Feature engineering complete!")
print("Saved as: earthquake_featured.csv")
