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


if __name__ == "__main__":
    df = load_raw()
    print(df.shape)
    print(df["A1Cresult"].value_counts(dropna=False))