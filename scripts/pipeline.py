"""Train and evaluate readmission models with patient-grouped validation."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, cross_val_predict
from sklearn.metrics import precision_recall_curve, confusion_matrix
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RANDOM_STATE = 42
THRESHOLD = 0.5   # fixed on OOF predictions; with scale_pos_weight it implies FN cost ≈ 7.8× FP

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


def get_oof_proba(model, X, y, groups):
    """Out-of-fold probabilities: each row is scored by a model that never saw its patient."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=-1,
                             method="predict_proba")[:, 1]


def tune_threshold(y, proba, beta=1):
    """Threshold that maximizes F-beta on out-of-fold predictions (beta > 1 favors recall)."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    b2 = beta ** 2
    f = (1 + b2) * precision * recall / (b2 * precision + recall + 1e-12)
    return thresholds[np.argmax(f[:-1])]   # last PR point has no threshold


def summarize_threshold(y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"recall": tp / (tp + fn), "precision": tp / (tp + fp),
            "fpr": fp / (fp + tn), "flagged": pred.mean()}


def add_fairness_groups(df):
    """Audit groups: merge small race groups, band age midpoints into three ranges."""
    return pd.DataFrame({
        "race": df["race"].replace({"Asian": "Other", "Hispanic": "Other"}),
        "gender": df["gender"],
        "age_band": pd.cut(df["age"], bins=[0, 50, 70, 100], right=False,
                           labels=["<50", "50-69", "70+"]),
    })


def fairness_table(y, pred, groups_df):
    """FNR (= 1 - recall) and FPR for every level of every audit attribute."""
    d = groups_df.assign(y=y.to_numpy(), pred=pred)
    rows = []
    for attr in groups_df.columns:
        for level, g in d.groupby(attr, observed=True):
            tp = ((g.y == 1) & (g.pred == 1)).sum()
            fn = ((g.y == 1) & (g.pred == 0)).sum()
            fp = ((g.y == 0) & (g.pred == 1)).sum()
            tn = ((g.y == 0) & (g.pred == 0)).sum()
            rows.append({"attribute": attr, "group": level, "n": len(g),
                         "readmit_rate": g.y.mean(), "FNR": fn / (tp + fn),
                         "FPR": fp / (fp + tn), "flagged": g.pred.mean()})
    return pd.DataFrame(rows)


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

    print("\nXGBoost threshold tuning (out-of-fold predictions on train):")
    xgb_proba = get_oof_proba(make_xgboost(num_cols, cat_cols, pos_weight), X, y, groups)
    best_t = tune_threshold(y, xgb_proba)
    for name, t in [("default", 0.5), ("F1-optimal", best_t),
                    ("F2-optimal", tune_threshold(y, xgb_proba, beta=2))]:
        s = summarize_threshold(y, xgb_proba, t)
        print(f"  {name:11s} t={t:.3f}  recall={s['recall']:.3f}  precision={s['precision']:.3f}"
              f"  FPR={s['fpr']:.3f}  flagged={s['flagged']:.1%}")

    print(f"\nFairness audit (XGBoost, out-of-fold, threshold={THRESHOLD}):")
    xgb_pred = (xgb_proba >= THRESHOLD).astype(int)
    table = fairness_table(y, xgb_pred, add_fairness_groups(train))
    print(table.round(3).to_string(index=False))
    table.to_csv(ROOT / "reports" / "fairness_oof.csv", index=False)

    gaps = table.groupby("attribute")[["FNR", "FPR"]].agg(lambda s: s.max() - s.min())
    print("\nLargest gap within each attribute (flag if > 0.10):")
    print(gaps.round(3))