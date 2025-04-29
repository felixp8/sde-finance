"""Merge multiple single-stock CSV files into one DataFrame with timestamp alignment."""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("../../data/full_history")
cutoff_date = pd.Timestamp("2011-01-01", tz="UTC")
keep_columns = ["Close", "Volume", "Open", "High", "Low", "Scaled_sentiment"]

# larger merged file
csv_files = sorted(DATA_DIR.glob("*.csv"))  # large
save_path = DATA_DIR / "../merged.csv"

# smaller merged file
# csv_files = [  # small
#     DATA_DIR / 'AMD.csv',
#     DATA_DIR / 'GOOG.csv',
#     DATA_DIR / 'KO.csv',
#     DATA_DIR / 'TSM.csv',
#     DATA_DIR / 'WMT.csv',
# ]
# save_path = DATA_DIR / "../merged_small.csv"

merged_df = []
for csv_file in csv_files:
    stock_name = csv_file.stem
    df = pd.read_csv(csv_file)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"{stock_name}: {df["Date"].iloc[0]}")
    if df["Date"].iloc[0] > cutoff_date:
        continue
    df = df[df["Date"] >= cutoff_date]
    df = df.set_index("Date")
    df = df[keep_columns]
    df.columns = [f'{stock_name}_{col}' for col in df.columns]
    merged_df.append(df)

merged_df = pd.concat(merged_df, axis=1)
print(merged_df.columns)
print(merged_df.isna().sum().sum())

merged_df.to_csv(save_path)