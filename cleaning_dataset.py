import pandas as pd

df = pd.read_excel("Earthquake_data_processed.xlsx")

# Drop unnamed
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

rename_map = {
    'Date(YYYY/MM/DD)': 'date',
    'Time(UTC)': 'time',
    'Latitude(deg)': 'latitude',
    'Longitude(deg)': 'longitude',
    'Depth(km)': 'depth_km',
    'Magnitude(ergs)': 'magnitude',
    'Magnitude_type': 'magnitude_type',
    'No_of_Stations': 'nst',
    'Gap': 'gap',
    'Close': 'close',
    'RMS': 'rms',
    'SRC': 'src',
    'EventID': 'eventid'
}

df = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns})

df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour

numeric_cols = ["latitude","longitude","depth_km","magnitude","nst","gap","close","rms"]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["latitude","longitude","depth_km","magnitude"])
df["sev_label"] = (df["magnitude"] >= 4).astype(int)

df.to_csv("earthquake_cleaned.csv", index=False)
print("DONE: earthquake_cleaned.csv generated!")
