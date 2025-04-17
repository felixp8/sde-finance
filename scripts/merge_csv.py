import pandas as pd
from pathlib import Path

DATA_DIR = Path("../../data/full_history")
csv_files = sorted(DATA_DIR.glob("*.csv"))
cutoff_date = pd.Timestamp("2011-01-01", tz="UTC")

merged_df = []
for csv_file in csv_files:
    stock_name = csv_file.stem
    df = pd.read_csv(csv_file)
    df["Date"] = pd.to_datetime(df["Date"])
    if df["Date"].iloc[0] > cutoff_date:
        continue
    df = df[df["Date"] >= cutoff_date]
    df = df.set_index("Date")
    df.columns = [f'{stock_name}_{col}' for col in df.columns]
    merged_df.append(df)

merged_df = pd.concat(merged_df, axis=1)
print(merged_df.columns)
print(merged_df.isna().sum().sum())

merged_df.to_csv(DATA_DIR / "../merged.csv")