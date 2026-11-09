"""Train and evaluate readmission models with patient-grouped validation."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RANDOM_STATE = 42

TARGET = "readmit_30d"
GROUP = "patient_nbr"
NOT_FEATURES = [TARGET, GROUP, "race"]   # race is kept only for the fairness audit
NUM_COLS = [
    "age", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]


def load_clean():
    return pd.read_csv(PROCESSED_DIR / "readmission_clean.csv", low_memory=False)


def split_train_test(df):
    """Hold out ~20% of patients as a test set; no patient appears in both parts."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(df, df[TARGET], groups=df[GROUP]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def get_feature_lists(df):
    cat_cols = [c for c in df.columns if c not in NUM_COLS + NOT_FEATURES]
    return NUM_COLS, cat_cols


def make_preprocessor(num_cols, cat_cols):
    """Scaling and one-hot encoding; fit inside each CV fold, never on the whole data."""
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=100,          # rare levels -> one "infrequent" column
                              sparse_output=False), cat_cols),
    ])


def make_logistic(num_cols, cat_cols):
    """Baseline: L1-regularized logistic regression with balanced class weights."""
    return Pipeline([
        ("prep", make_preprocessor(num_cols, cat_cols)),
        ("clf", LogisticRegression(l1_ratio=1, solver="liblinear", C=0.1,
                                   class_weight="balanced", max_iter=1000)),
    ])

def make_xgboost(num_cols, cat_cols, pos_weight):
    """Gradient-boosted trees; class imbalance handled with scale_pos_weight."""
    return Pipeline([
        ("prep", make_preprocessor(num_cols, cat_cols)),
        ("clf", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                              subsample=0.8, colsample_bytree=0.8,
                              scale_pos_weight=pos_weight, eval_metric="logloss",
                              n_jobs=1, random_state=RANDOM_STATE)),
    ])


def evaluate_cv(model, X, y, groups):
    """5-fold patient-grouped CV on the training set only."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X, y, groups=groups, cv=cv, n_jobs=-1,
                            scoring=["roc_auc", "average_precision", "recall"])
    return {m: scores[f"test_{m}"] for m in ["roc_auc", "average_precision", "recall"]}


if __name__ == "__main__":
    df = load_clean()
    train, test = split_train_test(df)
    print(f"Train: {len(train):,} rows, {train[GROUP].nunique():,} patients")
    print(f"Test:  {len(test):,} rows, {test[GROUP].nunique():,} patients")
    print(f"Patients in both train and test: {len(set(train[GROUP]) & set(test[GROUP]))}")

    num_cols, cat_cols = get_feature_lists(train)
    X, y, groups = train[num_cols + cat_cols], train[TARGET], train[GROUP]

    print("\nLogistic regression baseline (5-fold grouped CV on train):")
    results = evaluate_cv(make_logistic(num_cols, cat_cols), X, y, groups)
    for metric, values in results.items():
        print(f"  {metric:18s} mean={values.mean():.3f}  folds={values.round(3)}")

    pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"\nXGBoost (scale_pos_weight={pos_weight:.2f}, 5-fold grouped CV on train):")
    results = evaluate_cv(make_xgboost(num_cols, cat_cols, pos_weight), X, y, groups)
    for metric, values in results.items():
        print(f"  {metric:18s} mean={values.mean():.3f}  folds={values.round(3)}")