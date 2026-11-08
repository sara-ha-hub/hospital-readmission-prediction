"""Row-level cleaning of the UCI Diabetes 130-US Hospitals data.

Only fixed rules live here (no statistics learned from the data), so this
can safely run before the train/test split. Anything learned from the data
(scaling, encoding, feature selection, resampling) goes inside the
model pipeline instead.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


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


DROP_COLS = [
    "encounter_id",              # just an ID
    "weight",                    # 97% missing
    "payer_code",                # 40% missing, not clinical
    "readmitted",                # replaced by readmit_30d
    "examide", "citoglipton",    # always "No", no information
]
CODE_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]


def clean_columns(df):
    df = df.drop(columns=DROP_COLS)
    df[CODE_COLS] = df[CODE_COLS].astype(str)                       # codes are labels, not numbers
    df["race"] = df["race"].fillna("Unknown")
    df["medical_specialty"] = df["medical_specialty"].fillna("Missing")
    return df


DIAG_COLS = ["diag_1", "diag_2", "diag_3"]


def icd9_group(code):
    """Map one ICD-9 code to a broad clinical group (Strack et al., 2014)."""
    if pd.isna(code):
        return "Missing"
    code = str(code)
    if code.startswith(("V", "E")):          # supplementary / external-cause codes
        return "Other"
    v = int(float(code))                     # use the whole-number part, e.g. 250.83 -> 250
    if v == 250:                     return "Diabetes"
    if 390 <= v <= 459 or v == 785:  return "Circulatory"
    if 460 <= v <= 519 or v == 786:  return "Respiratory"
    if 520 <= v <= 579 or v == 787:  return "Digestive"
    if 580 <= v <= 629 or v == 788:  return "Genitourinary"
    if 710 <= v <= 739:              return "Musculoskeletal"
    if 800 <= v <= 999:              return "Injury"
    if 140 <= v <= 239:              return "Neoplasms"
    return "Other"


def group_diagnoses(df):
    for col in DIAG_COLS:
        df[col] = df[col].map(icd9_group)
    return df


RARE_MEDS = [                    # each prescribed in fewer than 0.1% of encounters
    "chlorpropamide", "acetohexamide", "tolbutamide", "miglitol",
    "troglitazone", "tolazamide", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]
AGE_MIDPOINT = {f"[{i}-{i + 10})": i + 5 for i in range(0, 100, 10)}  # "[70-80)" -> 75


def finalize(df):
    df = df.drop(columns=RARE_MEDS)
    df["age"] = df["age"].map(AGE_MIDPOINT)                         # ordered bands -> number
    return df


def build_clean_data():
    df = load_raw()
    df = filter_rows(df)
    df = clean_columns(df)
    df = group_diagnoses(df)
    df = finalize(df)
    return df


if __name__ == "__main__":
    df = build_clean_data()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "readmission_clean.csv"
    df.to_csv(out_path, index=False)

    print(f"Shape: {df.shape}")
    print(f"Patients: {df['patient_nbr'].nunique():,}")
    print(f"Readmit <30d rate: {df['readmit_30d'].mean():.1%}")
    print(f"Missing values left: {df.isna().sum().sum()}")
    print(f"Saved -> {out_path.relative_to(ROOT)}")