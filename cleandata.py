import pandas as pd

df = pd.read_csv("data/class5data.csv")

df.columns = df.columns.str.strip()

for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})

if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female", "fe male": "Female", "male ": "Male", "female ": "Female"})
    df["Gender"] = df["Gender"].replace({"M": "Male", "F": "Female"})

if "ProdTaken" in df.columns:
    df["ProdTaken"] = pd.to_numeric(df["ProdTaken"], errors="coerce").astype("Int64")

for c in df.columns:
    if c != "ProdTaken" and df[c].dtype == "object":
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().sum() >= int(0.8 * len(df)):
            df[c] = converted

num_cols = df.select_dtypes(include=["int64", "float64", "Int64"]).columns.tolist()
obj_cols = df.select_dtypes(include=["object"]).columns.tolist()

for c in num_cols:
    med = df[c].median()
    df[c] = df[c].fillna(med)

for c in obj_cols:
    mode = df[c].mode(dropna=True)
    fill = mode.iloc[0] if len(mode) > 0 else "Unknown"
    df[c] = df[c].fillna(fill)

df.to_csv("data/class5data_clean.csv", index=False)

print(df.shape)
missing = df.isna().sum()
print(missing[missing > 0].sort_values(ascending=False))