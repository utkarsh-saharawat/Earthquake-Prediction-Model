"""
USGS-style models for earthquake zone classification, ETAS forecasting,
and nowcast scoring — fully vectorized & compatible with app.py and evaluation scripts.
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, atan2, sqrt

# =====================================================================
#                       VECTOR-SAFE HAVERSINE
# =====================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    """
    Computes haversine great-circle distance (in km).

    Supports:
    - scalars
    - numpy arrays
    - broadcasting (scalar vs array)
    """

    R = 6371.0  # Earth radius in km

    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    # Convert to radians
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2)**2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


# =====================================================================
#                ZONE CLASSIFIER (1–5) — USGS STYLE
# =====================================================================
def classify_zone(
    lat, lon, catalog,
    radius_km=200.0,
    lookback_days=365,
    thresholds=None
):
    """
    Classify a location (lat, lon) into seismic hazard Zone 1–5.

    catalog must include:
    - datetime (pd.Timestamp)
    - latitude
    - longitude
    - magnitude

    Output:
    {
        "zone": int (1..5),
        "score": float,
        "reason": str,
        ...
    }
    """

    # Default USGS heuristic thresholds
    if thresholds is None:
        thresholds = {
            "events_30d_high": 50,
            "events_365_high": 200,
            "max_mag_30d_vhigh": 6.0,
            "max_mag_365_vhigh": 7.0,
            "energy_30d_high": 1e9,
            "b_low": 0.6,
        }

    # Ensure datetime format
    if not np.issubdtype(catalog["datetime"].dtype, np.datetime64):
        catalog = catalog.copy()
        catalog["datetime"] = pd.to_datetime(catalog["datetime"], errors="coerce")

    anchor = catalog["datetime"].max()
    if pd.isnull(anchor):
        anchor = pd.Timestamp.now()

    # ---------------------------
    # Extract local events (within radius)
    # ---------------------------
    dists = haversine_km(lat, lon, catalog["latitude"].values, catalog["longitude"].values)
    local = catalog[dists <= radius_km].copy()

    # Time windows
    w30 = anchor - pd.Timedelta(days=30)
    w365 = anchor - pd.Timedelta(days=365)

    local_30 = local[local["datetime"] >= w30]
    local_365 = local[local["datetime"] >= w365]

    events_30 = len(local_30)
    events_365 = len(local_365)
    max30 = float(local_30["magnitude"].max()) if len(local_30) else 0.0
    max365 = float(local_365["magnitude"].max()) if len(local_365) else 0.0

    # ---------------------------
    # Energy proxy
    # ---------------------------
    def energy(df):
        if len(df) == 0:
            return 0.0
        return float((10 ** (1.5 * df["magnitude"])).sum())

    energy30 = energy(local_30)

    # ---------------------------
    # b-value estimate
    # ---------------------------
    b_val = None
    if len(local_365) >= 5:
        mags = local_365["magnitude"].dropna()
        mmin = mags.min()
        mmean = mags.mean()
        if (mmean - mmin) > 0:
            b_val = 1.0 / ((mmean - mmin) * np.log(10))

    # ---------------------------
    # SCORING SYSTEM (0–9)
    # ---------------------------
    score = 0.0
    reasons = []

    # Strong recent quake
    if max30 >= thresholds["max_mag_30d_vhigh"]:
        score += 3.0
        reasons.append(f"Recent M {max30:.1f} in last 30d")

    if max365 >= thresholds["max_mag_365_vhigh"]:
        score += 3.0
        reasons.append(f"Recent M {max365:.1f} in last 365d")

    # Event counts
    if events_30 >= thresholds["events_30d_high"]:
        score += 2.5
        reasons.append(f"{events_30} events in 30d (high)")
    elif events_30 >= thresholds["events_30d_high"] / 2:
        score += 1.0
        reasons.append(f"{events_30} events in 30d (moderate)")

    if events_365 >= thresholds["events_365_high"]:
        score += 2.0
        reasons.append(f"{events_365} events in 365d")

    # Energy
    if energy30 >= thresholds["energy_30d_high"]:
        score += 2.0
        reasons.append("High total seismic energy (30d)")

    # b-value
    if b_val is not None and b_val < thresholds["b_low"]:
        score += 1.5
        reasons.append(f"Low b-value {b_val:.2f}")

    # Nearby M>=6 events
    big = local_365[local_365["magnitude"] >= 6.0]
    if len(big):
        big_dists = haversine_km(lat, lon, big["latitude"].values, big["longitude"].values)
        dmin = float(big_dists.min())
        if dmin < 100:
            score += 2.0
            reasons.append(f"M6+ within {dmin:.0f} km")

    # ---------------------------
    # MAP SCORE → ZONE
    # ---------------------------
    if score >= 6.0:
        zone = 5
    elif score >= 4.0:
        zone = 4
    elif score >= 2.0:
        zone = 3
    elif score >= 1.0:
        zone = 2
    else:
        zone = 1

    reason_text = "; ".join(reasons) if reasons else "Low seismicity"

    return {
        "zone": zone,
        "score": float(score),
        "reason": reason_text,
        "events_30": int(events_30),
        "events_365": int(events_365),
        "max30": float(max30),
        "max365": float(max365),
        "energy30": float(energy30),
        "b_value": float(b_val) if b_val is not None else None
    }


# =====================================================================
#                        SIMPLE ETAS FORECAST
# =====================================================================
def etas_forecast(catalog, target_time, window_days=1.0, params=None):
    """
    Simple temporal ETAS model (non-spatial).
    """

    if params is None:
        params = {'mu': 0.01, 'K': 0.5, 'alpha': 1.0, 'c': 0.01, 'p': 1.1}

    mu = params['mu']
    K = params['K']
    alpha = params['alpha']
    c = params['c']
    p = params['p']

    # Filter past events
    hist = catalog[catalog["datetime"] < target_time].copy()
    if hist.empty:
        return {
            'expected_total': mu * window_days,
            'expected_background': mu * window_days,
            'expected_triggered': 0.0,
            'by_event': np.array([]),
            'params': params
        }

    # Convert times
    T0 = target_time.timestamp()
    t_i = (hist['datetime'].astype('int64') // 10**9).astype(float)
    dt = T0 - t_i
    W = window_days * 86400.0

    mags = hist["magnitude"].fillna(hist["magnitude"].median()).values
    M0 = 2.5
    prod = K * np.exp(alpha * (mags - M0))

    # Omori temporal integral
    if abs(p - 1.0) > 1e-6:
        int_temp = (((dt + W + c)**(1 - p) - (dt + c)**(1 - p)) / (1 - p))
    else:
        int_temp = np.log((dt + W + c) / (dt + c))

    by_event = prod * int_temp
    expected_triggered = float(by_event.sum())
    expected_background = mu * window_days
    expected_total = expected_triggered + expected_background

    return {
        'expected_total': expected_total,
        'expected_background': expected_background,
        'expected_triggered': expected_triggered,
        'by_event': by_event,
        'params': params
    }


# =====================================================================
#                              NOWCAST
# =====================================================================
def nowcast_score(catalog, origin_time, lat, lon, R_km=300):
    """
    Simple USGS-style nowcast score (0–1 scale).
    Counts number of events in 1d, 7d, 30d, 365d
    """

    if not np.issubdtype(catalog["datetime"].dtype, np.datetime64):
        catalog = catalog.copy()
        catalog["datetime"] = pd.to_datetime(catalog["datetime"], errors="coerce")

    hist = catalog[catalog["datetime"] <= origin_time].copy()
    if hist.empty:
        return {"score": 0.0, "counts": {}}

    dists = haversine_km(lat, lon, hist["latitude"].values, hist["longitude"].values)
    hist = hist[dists <= R_km]
    if hist.empty:
        return {"score": 0.0, "counts": {}}

    w1 = origin_time - pd.Timedelta(days=1)
    w7 = origin_time - pd.Timedelta(days=7)
    w30 = origin_time - pd.Timedelta(days=30)
    w365 = origin_time - pd.Timedelta(days=365)

    c1 = len(hist[hist["datetime"] >= w1])
    c7 = len(hist[hist["datetime"] >= w7])
    c30 = len(hist[hist["datetime"] >= w30])
    c365 = len(hist[hist["datetime"] >= w365])

    score = 0.0
    score += min(c1 / 10, 1.0) * 0.4
    score += min(c7 / 50, 1.0) * 0.3
    score += min(c30 / 200, 1.0) * 0.2
    score += min(c365 / 1000, 1.0) * 0.1

    return {
        "score": float(score),
        "counts": {"1d": c1, "7d": c7, "30d": c30, "365d": c365}
    }
