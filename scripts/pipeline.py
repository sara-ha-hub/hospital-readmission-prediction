"""Train and evaluate readmission models with patient-grouped validation."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RANDOM_STATE = 42

TARGET = "readmit_30d"
GROUP = "patient_nbr"


def load_clean():
    return pd.read_csv(PROCESSED_DIR / "readmission_clean.csv", low_memory=False)


def split_train_test(df):
    """Hold out ~20% of patients as a test set; no patient appears in both parts."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(df, df[TARGET], groups=df[GROUP]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


if __name__ == "__main__":
    df = load_clean()
    train, test = split_train_test(df)

    shared = set(train[GROUP]) & set(test[GROUP])
    print(f"Train: {len(train):,} rows, {train[GROUP].nunique():,} patients, "
          f"readmit rate {train[TARGET].mean():.1%}")
    print(f"Test:  {len(test):,} rows, {test[GROUP].nunique():,} patients, "
          f"readmit rate {test[TARGET].mean():.1%}")
    print(f"Patients in both train and test: {len(shared)}")