from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_raw():
    df = pd.read_csv(
        RAW_DIR / "diabetic_data.csv",
        na_values=["?"],
        keep_default_na=False,     # keep "None" as a real value (= test not done), not missing
        low_memory=False,
    )
    return df

EXPIRED_HOSPICE = [11, 13, 14, 19, 20, 21]  # discharge codes: died or sent to hospice


def filter_rows(df):
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_HOSPICE)]  # cannot be readmitted
    df = df[df["gender"] != "Unknown/Invalid"]                       # only 3 rows
    df = df.copy()
    df["readmit_30d"] = (df["readmitted"] == "<30").astype(int)      # target: 1 = back within 30 days
    return df

if __name__ == "__main__":
    df = load_raw()
    df = filter_rows(df)
    print(df.shape)
    print(f"Patients: {df['patient_nbr'].nunique():,}")
    print(f"Readmit <30d rate: {df['readmit_30d'].mean():.1%}")