"""
IntelliCrash Hardware — TOPSIS Decision Analysis & Best Model Export
====================================================================
Reads consolidated results, executes TOPSIS ranking strictly on validation metrics,
identifies the Rank #1 winning model, generates its evaluation report, and exports
its deployment configuration.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

SEED = 42
KEPT_PER_BLOCK = 14
HARDWARE_DIR = Path(__file__).resolve().parent
MODELS_DIR = HARDWARE_DIR / "models"
PREDS_PATH = MODELS_DIR / "test_predictions.npz"


def _topsis_scores(data, weights, impacts):
    data = np.asarray(data, dtype=float)
    rss = np.sqrt((data**2).sum(axis=0))
    rss[rss == 0] = 1e-9
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    wd = data / rss * w
    plus = np.array([i == "+" for i in impacts])
    best = np.where(plus, wd.max(axis=0), wd.min(axis=0))
    worst = np.where(plus, wd.min(axis=0), wd.max(axis=0))
    d_b = np.sqrt(((wd - best) ** 2).sum(axis=1))
    d_w = np.sqrt(((wd - worst) ** 2).sum(axis=1))
    den = d_b + d_w
    den[den == 0] = 1e-9
    return d_w / den


def _pred_key(name):
    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def _load_preds(name):
    if not PREDS_PATH.exists():
        return None
    with np.load(PREDS_PATH) as d:
        k = _pred_key(name)
        return d[k].astype(int) if k in d.files else None


def main():
    print("=" * 75)
    print("  STANDALONE TOPSIS DECISION ANALYSIS & BEST MODEL EXPORT")
    print("=" * 75)

    cons_path = MODELS_DIR / "consolidated_results_table.csv"
    if not cons_path.exists():
        print(f"  [ERROR] {cons_path} not found. Please run `train_models.py` first.")
        return

    cons = pd.read_csv(cons_path)

    # TOPSIS Criteria Weights on Validation Metrics
    val_cols = [
        "Val_Accuracy",
        "Val_Precision",
        "Val_Recall",
        "Val_F1-Score",
        "Val_AUC-ROC",
    ]
    weights = [0.05, 0.10, 0.40, 0.20, 0.25]
    impacts = ["+", "+", "+", "+", "+"]

    topsis_in = cons[["Model / Pipeline Stage"] + val_cols].copy()
    data = topsis_in[val_cols].to_numpy(dtype=float)
    scores = _topsis_scores(data, weights, impacts)

    cons["Topsis Score"] = scores
    cons["Rank"] = pd.Series(scores).rank(ascending=False, method="dense").astype(int)
    res = cons.sort_values("Rank").reset_index(drop=True)

    print(
        f"\n  {'Rank':>4}  {'Model / Pipeline Stage':<45} {'ValRec':>7} {'ValAUC':>7} | {'TestRec':>7} {'TestF1':>7} {'Ci':>8}"
    )
    print(f"  {'-' * 105}")
    for _, r in res.iterrows():
        print(
            f"  {int(r['Rank']):>4}  {r['Model / Pipeline Stage']:<45} {r['Val_Recall']:>7.4f} {r['Val_AUC-ROC']:>7.4f} | "
            f"{r['Recall']:>7.4f} {r['F1-Score']:>7.4f} {r['Topsis Score']:>8.4f}"
        )

    # Identify Rank #1 Winner
    best_row = res.loc[res["Rank"] == 1].iloc[0]
    best_name = best_row["Model / Pipeline Stage"]
    best_score = float(best_row["Topsis Score"])

    print(f"\n" + "=" * 75)
    print(f"  ★ TOPSIS RANK #1 WINNING MODEL SELECTED: {best_name}")
    print(f"    Closeness Score (Ci): {best_score:.4f}")
    print("=" * 75)

    # Save Rankings Table
    ranking_csv = MODELS_DIR / "topsis_hardware_ranking.csv"
    res.to_csv(ranking_csv, index=False)
    print(f"  [OK] Saved full rankings table: {ranking_csv}")

    # Export & Generate Deep Evaluation Report for Rank #1 Model
    d = np.load(HARDWARE_DIR / "data" / "processed_splits.npz")
    y_test = d["test_y"]
    y_pred = _load_preds(best_name)

    if y_pred is not None and len(y_pred) == len(y_test):
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        report = classification_report(
            y_test, y_pred, target_names=["NON_CRASH", "CRASH"], zero_division=0
        )

        print(f"\n  Confusion Matrix ({best_name}):")
        print(f"                    Predicted")
        print(f"                    NON_CRASH  CRASH")
        print(f"  Actual NON_CRASH  {cm[0][0]:>6}   {cm[0][1]:>6}")
        print(f"  Actual CRASH      {cm[1][0]:>6}   {cm[1][1]:>6}")

        eval_txt = MODELS_DIR / "best_model_evaluation.txt"
        with open(eval_txt, "w") as f:
            f.write(
                f"TOPSIS Rank #1 Winning Model (Selected on Validation): {best_name}\n"
            )
            f.write(f"TOPSIS Closeness Score (Ci): {best_score:.4f}\n\n")
            f.write("Test Set Confusion Matrix:\n")
            f.write(
                "                    Predicted\n                    NON_CRASH  CRASH\n"
            )
            f.write(f"  Actual NON_CRASH  {cm[0][0]:>6}   {cm[0][1]:>6}\n")
            f.write(f"  Actual CRASH      {cm[1][0]:>6}   {cm[1][1]:>6}\n\n")
            f.write(f"Classification Report:\n{report}")
        print(f"  [OK] Exported Rank #1 evaluation report: {eval_txt}")
    else:
        print(
            "  [INFO] Test predictions cache not found or shape mismatch for detailed confusion matrix export."
        )

    # Save Winning Model Metadata JSON
    winner_json = MODELS_DIR / "topsis_winner_metadata.json"
    import json

    with open(winner_json, "w") as f:
        json.dump(
            {
                "winning_model": best_name,
                "topsis_score": best_score,
                "val_recall": float(best_row["Val_Recall"]),
                "val_auc_roc": float(best_row["Val_AUC-ROC"]),
                "test_accuracy": float(best_row["Accuracy"]),
                "test_f1": float(best_row["F1-Score"]),
            },
            f,
            indent=2,
        )
    print(f"  [OK] Exported winner metadata: {winner_json}")
    print("  ✓ TOPSIS analysis & model export successfully completed!")


if __name__ == "__main__":
    main()
