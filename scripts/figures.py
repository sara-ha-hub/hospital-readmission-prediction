"""Figures for the mid-project report, built from the saved out-of-fold predictions."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)
from pipeline import PROCESSED_DIR, ROOT, TARGET, THRESHOLD, tune_threshold

FIG_DIR = ROOT / "reports" / "figures"
MODELS = {"Logistic (L1)": "lr_proba", "XGBoost": "xgb_proba"}


def plot_roc_pr(oof):
    """ROC and precision-recall curves for both models."""
    y = oof[TARGET]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, col in MODELS.items():
        fpr, tpr, _ = roc_curve(y, oof[col])
        ax1.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc_score(y, oof[col]):.3f})")
        precision, recall, _ = precision_recall_curve(y, oof[col])
        ax2.plot(recall, precision, label=f"{label} (AP = {average_precision_score(y, oof[col]):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax2.axhline(y.mean(), color="k", ls="--", lw=1, label=f"Base rate ({y.mean():.3f})")
    ax1.set(xlabel="False positive rate", ylabel="Recall (true positive rate)", title="ROC curve")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    ax1.legend(loc="lower right")
    ax2.legend(loc="upper right")
    fig.suptitle("Out-of-fold predictions, 5-fold patient-grouped CV on the training set")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "oof_roc_pr.png", dpi=200)
    plt.close(fig)


def plot_threshold_tradeoff(oof):
    """Recall, precision and share flagged across thresholds for XGBoost."""
    y, p = oof[TARGET].to_numpy(), oof["xgb_proba"].to_numpy()
    ts = np.linspace(0.2, 0.8, 121)   # outside this range almost nobody / everybody is flagged
    flagged = np.array([(p >= t).sum() for t in ts])
    hits = np.array([((p >= t) & (y == 1)).sum() for t in ts])
    recall = hits / (y == 1).sum()
    precision = np.divide(hits, flagged, out=np.full(len(ts), np.nan), where=flagged > 0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ts, recall, label="Recall")
    ax.plot(ts, precision, label="Precision")
    ax.plot(ts, flagged / len(p), label="Share of encounters flagged")
    marks = {"F2-optimal": tune_threshold(y, p, beta=2), "Chosen": THRESHOLD,
             "F1-optimal": tune_threshold(y, p)}
    align = {"F2-optimal": "center", "Chosen": "right", "F1-optimal": "left"}
    for name, t in marks.items():
        ax.axvline(t, color="k", lw=1.5 if name == "Chosen" else 0.8,
                   ls="-" if name == "Chosen" else ":")
        ax.text(t, 1.02, f"{name}\n{t:.3f}", ha=align[name], va="bottom", fontsize=8)
    ax.set(xlabel="Decision threshold", ylabel="Rate", ylim=(0, 1), xlim=(0.2, 0.8))
    ax.set_title("XGBoost threshold trade-off (out-of-fold)", pad=28)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "oof_threshold_tradeoff.png", dpi=200)
    plt.close(fig)


def plot_fairness():
    """FNR and FPR per audit group, read from the table pipeline.py saved."""
    table = pd.read_csv(ROOT / "reports" / "fairness_oof.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True,
                             gridspec_kw={"width_ratios": [4, 2, 3]})
    for ax, (attr, g) in zip(axes, table.groupby("attribute", sort=False)):
        x, w = np.arange(len(g)), 0.38
        ax.bar(x - w / 2, g["FNR"], w, label="FNR (missed readmissions)")
        ax.bar(x + w / 2, g["FPR"], w, label="FPR (false alarms)")
        ax.set_xticks(x, [f"{grp}\n(n={n:,})" for grp, n in zip(g["group"], g["n"])], fontsize=8)
        ax.set_title(attr)
    axes[0].set_ylabel("Rate")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"XGBoost error rates by group (out-of-fold, threshold = {THRESHOLD})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fairness_fnr_fpr.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    oof = pd.read_csv(PROCESSED_DIR / "oof_predictions.csv")
    plot_roc_pr(oof)
    plot_threshold_tradeoff(oof)
    plot_fairness()
    print(f"Saved 3 figures to {FIG_DIR}")