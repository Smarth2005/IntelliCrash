"""
IntelliCrash Hardware — Training, Evaluation & Edge Export Pipeline
===================================================================
1. Loads preprocessed splits from preprocessing.py
2. Trains Classical ML, GBDTs (XGBoost, LightGBM), and Deep Learning baselines
3. Trains the multi-task Dual-Head Bi-LSTM using Focal Loss and MSE Loss
4. Optimizes Hybrid Bimodal (Bi-LSTM + CSI) and Trimodal fusion weights
5. Saves PyTorch checkpoints (.pth) and exports ONNX models (.onnx) for Raspberry Pi
6. Compiles and saves `consolidated_results_table.csv`
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import sys
import json
from pathlib import Path

HARDWARE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HARDWARE_DIR.parent
sys.path.insert(0, str(HARDWARE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from bilstm import IntelliCrashBiLSTM, IMUDataset, FocalLoss
    from feature_engineering import NUM_ENGINEERED_FEATURES
except ImportError:
    from hardware.bilstm import IntelliCrashBiLSTM, IMUDataset, FocalLoss
    from hardware.feature_engineering import NUM_ENGINEERED_FEATURES

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

SEED = 42
LSTM_INPUT_SIZE = 6 + NUM_ENGINEERED_FEATURES
THRESH_BETA = 1.5
MODELS_DIR = HARDWARE_DIR / "models"
PREDS_PATH = MODELS_DIR / "test_predictions.npz"


def set_all_seeds(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fbeta(y, preds, beta=THRESH_BETA):
    p = precision_score(y, preds, zero_division=0)
    r = recall_score(y, preds, zero_division=0)
    d = beta**2 * p + r
    return (1 + beta**2) * p * r / d if d > 0 else 0.0


def _tune_threshold(y_val, val_probs, beta=THRESH_BETA):
    val_probs = np.asarray(val_probs, dtype=float)
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.10, 0.90, 81):
        s = _fbeta(y_val, (val_probs > t).astype(int), beta)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t


def _basic_metrics(y, probs, thr=0.5):
    probs = np.asarray(probs, dtype=float)
    preds = (probs > thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "auc_roc": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.0,
    }


def _save_preds(name, preds):
    store = {}
    if PREDS_PATH.exists():
        with np.load(PREDS_PATH) as d:
            store = {k: d[k] for k in d.files}
    import re

    key = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    store[key] = np.asarray(preds).astype(np.int8)
    PREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PREDS_PATH, **store)


def _evaluate_model(name, y_val, val_probs, y_test, test_probs, thr=None):
    if thr is None:
        thr = _tune_threshold(y_val, val_probs)
    m = _basic_metrics(y_test, test_probs, thr)
    v = _basic_metrics(y_val, val_probs, thr)
    m.update({"val_" + k: x for k, x in v.items()})
    m["threshold"] = float(thr)
    _save_preds(name, (np.asarray(test_probs) > thr).astype(int))
    return m


def build_lstm_input(X_raw, features):
    seq_len = X_raw.shape[1]
    features_broadcast = np.repeat(features[:, np.newaxis, :], seq_len, axis=1)
    return np.concatenate([X_raw, features_broadcast], axis=2).astype(np.float32)


def export_for_edge_deployment(model, hparams, device):
    """Saves PyTorch weights and exports ONNX model for Raspberry Pi deployment."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save PyTorch Checkpoint
    ckpt_path = MODELS_DIR / "hw_best_bilstm.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hparams": {
                k: hparams[k] for k in ("hidden_size", "num_layers", "dropout")
            },
        },
        ckpt_path,
    )
    print(f"  [OK] Saved PyTorch checkpoint: {ckpt_path}")

    # 2. Export to ONNX for Raspberry Pi (ARM compatibility via onnxruntime)
    onnx_path = MODELS_DIR / "hw_bilstm.onnx"
    model.eval()
    dummy_input = torch.randn(1, 200, LSTM_INPUT_SIZE).to(device)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_telemetry"],
            output_names=["crash_probability", "severity_score"],
            dynamic_axes={
                "input_telemetry": {0: "batch_size"},
                "crash_probability": {0: "batch_size"},
                "severity_score": {0: "batch_size"},
            },
        )
        print(
            f"  [OK] Exported ONNX model for Raspberry Pi: {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)"
        )
    except Exception as e:
        print(f"  [WARN] ONNX export notice: {e}")


def run_training_pipeline():
    print("=" * 70)
    print("  MODEL TRAINING, FUSION OPTIMIZATION & EDGE EXPORT PIPELINE")
    print("=" * 70)

    set_all_seeds(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Execution Device: {device}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load preprocessed splits
    splits_path = HARDWARE_DIR / "data" / "processed_splits.npz"
    if not splits_path.exists():
        print(
            f"  [ERROR] {splits_path} not found. Please run `preprocessing.py` first."
        )
        return

    d = np.load(splits_path)
    splits = {
        "train": {
            "X": d["train_X"],
            "y": d["train_y"],
            "features": d["train_feat"],
            "csi": d["train_csi"],
        },
        "val": {
            "X": d["val_X"],
            "y": d["val_y"],
            "features": d["val_feat"],
            "csi": d["val_csi"],
        },
        "test": {
            "X": d["test_X"],
            "y": d["test_y"],
            "features": d["test_feat"],
            "csi": d["test_csi"],
        },
    }

    y_val, y_test = splits["val"]["y"], splits["test"]["y"]
    csi_val, csi_test = splits["val"]["csi"], splits["test"]["csi"]

    # 1. Classical & GBDT Baselines
    print("\n  [Stage 1] Training Classical & Tabular GBDT Baselines...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    import xgboost as xgb
    import lightgbm as lgb

    X_tr, X_v, X_te = (
        splits["train"]["features"],
        splits["val"]["features"],
        splits["test"]["features"],
    )
    y_tr = splits["train"]["y"]

    table1 = {}
    for name, clf in [
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=SEED, n_jobs=-1
            ),
        ),
        (
            "SVM (RBF Kernel)",
            SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED),
        ),
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=SEED)),
    ]:
        clf.fit(X_tr, y_tr)
        table1[name] = _evaluate_model(
            name,
            y_val,
            clf.predict_proba(X_v)[:, 1],
            y_test,
            clf.predict_proba(X_te)[:, 1],
        )

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, random_state=SEED
    )
    xgb_model.fit(X_tr, y_tr, verbose=False)
    xgb_test_probs = xgb_model.predict_proba(X_te)[:, 1]
    xgb_val_probs = xgb_model.predict_proba(X_v)[:, 1]
    table1["XGBoost (Fine-Tuned)"] = _evaluate_model(
        "XGBoost (Fine-Tuned)", y_val, xgb_val_probs, y_test, xgb_test_probs
    )

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, random_state=SEED, verbose=-1
    )
    lgb_model.fit(X_tr, y_tr)
    lgb_test_probs = lgb_model.predict_proba(X_te)[:, 1]
    lgb_val_probs = lgb_model.predict_proba(X_v)[:, 1]
    table1["LightGBM"] = _evaluate_model(
        "LightGBM", y_val, lgb_val_probs, y_test, lgb_test_probs
    )

    # 2. Train Multi-Task Bi-LSTM
    print("\n  [Stage 2] Training Multi-Task Bi-LSTM Architecture...")
    hparams = {
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "lr": 0.001,
        "batch_size": 64,
        "epochs": 30,
        "weight_decay": 0.01,
        "focal_alpha": 0.75,
        "focal_gamma": 2.0,
    }

    X_train_lstm = build_lstm_input(splits["train"]["X"], X_tr)
    X_val_lstm = build_lstm_input(splits["val"]["X"], X_v)
    X_test_lstm = build_lstm_input(splits["test"]["X"], X_te)

    train_ds = IMUDataset(X_train_lstm, y_tr, splits["train"]["csi"])
    train_loader = DataLoader(train_ds, batch_size=hparams["batch_size"], shuffle=True)

    model = IntelliCrashBiLSTM(
        input_size=LSTM_INPUT_SIZE,
        hidden_size=hparams["hidden_size"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"]
    )
    crit_crash = FocalLoss(alpha=hparams["focal_alpha"], gamma=hparams["focal_gamma"])
    crit_sev = nn.MSELoss()

    for epoch in range(hparams["epochs"]):
        model.train()
        for batch in train_loader:
            bx = batch["features"].to(device)
            by_c = batch["crash"].to(device).float().squeeze(-1)
            by_s = batch["severity"].to(device).float().squeeze(-1)

            optimizer.zero_grad()
            pc, ps = model(bx)
            loss = 0.7 * crit_crash(pc.squeeze(-1), by_c) + 0.3 * crit_sev(
                ps.squeeze(-1), by_s
            )
            loss.backward()
            optimizer.step()

    model.eval()

    def get_lstm_probs(X_in):
        ds = IMUDataset(X_in, np.zeros(len(X_in)), np.zeros(len(X_in)))
        loader = DataLoader(ds, batch_size=64)
        out = []
        with torch.no_grad():
            for b in loader:
                pc, _ = model(b["features"].to(device))
                out.extend(pc.cpu().numpy().flatten())
        return np.array(out)

    lstm_val_probs = get_lstm_probs(X_val_lstm)
    lstm_test_probs = get_lstm_probs(X_test_lstm)
    table1["Bi-LSTM Only (Deep Temporal)"] = _evaluate_model(
        "Bi-LSTM Only (Deep Temporal)", y_val, lstm_val_probs, y_test, lstm_test_probs
    )

    # Export Bi-LSTM Checkpoint & ONNX for Edge Deployment
    export_for_edge_deployment(model, hparams, device)

    # 3. Hybrid Bi-Modal Fusion (Bi-LSTM + CSI)
    print("\n  [Stage 3] Optimizing Hybrid Bimodal Fusion Weights...")
    best_w, best_wc, best_tau, best_score = 0.6, 0.4, 0.5, -1
    for w_l in np.linspace(0.5, 1.0, 11):
        w_c = round(1.0 - w_l, 2)
        p_v = w_l * lstm_val_probs + w_c * csi_val
        tau = _tune_threshold(y_val, p_v)
        score = f1_score(y_val, (p_v > tau).astype(int), zero_division=0)
        if score > best_score:
            best_score, best_w, best_wc, best_tau = score, w_l, w_c, tau

    fusion_name = f"Hybrid Fusion Gate (Optimized {best_w} : {best_wc})"
    table1[fusion_name] = _evaluate_model(
        fusion_name,
        y_val,
        best_w * lstm_val_probs + best_wc * csi_val,
        y_test,
        best_w * lstm_test_probs + best_wc * csi_test,
        thr=best_tau,
    )
    table1["CSI Only (Physics Path)"] = _evaluate_model(
        "CSI Only (Physics Path)", y_val, csi_val, y_test, csi_test
    )

    # Save Optimal Fusion Weights JSON for Edge Inference
    with open(MODELS_DIR / "optimal_fusion_weights.json", "w") as f:
        json.dump(
            {
                "w_ml": float(best_w),
                "w_csi": float(best_wc),
                "threshold": float(best_tau),
                "best_weights": f"{best_w} : {best_wc}",
            },
            f,
            indent=2,
        )

    # 4. Trimodal Fusion (Bi-LSTM + CSI + GBDT)
    print("\n  [Stage 4] Optimizing Trimodal Fusion Variants...")
    for tm_name, tab_v, tab_t in [
        ("Tri-Modal (BiLSTM + CSI + LightGBM)", lgb_val_probs, lgb_test_probs),
        ("Tri-Modal (BiLSTM + CSI + XGBoost)", xgb_val_probs, xgb_test_probs),
    ]:
        best_tm_score, best_params = -1, (0.4, 0.4, 0.2, 0.5)
        for w1 in np.linspace(0.2, 0.6, 9):
            for w2 in np.linspace(0.2, 0.6, 9):
                w3 = round(1.0 - w1 - w2, 2)
                if w3 >= 0.05:
                    p_v = w1 * lstm_val_probs + w2 * tab_v + w3 * csi_val
                    tau = _tune_threshold(y_val, p_v)
                    sc = f1_score(y_val, (p_v > tau).astype(int), zero_division=0)
                    if sc > best_tm_score:
                        best_tm_score, best_params = sc, (w1, w2, w3, tau)
        w1, w2, w3, tau = best_params
        table1[tm_name] = _evaluate_model(
            tm_name,
            y_val,
            w1 * lstm_val_probs + w2 * tab_v + w3 * csi_val,
            y_test,
            w1 * lstm_test_probs + w2 * tab_t + w3 * csi_test,
            thr=tau,
        )

    # 5. Compile and Save Consolidated Results Table CSV
    rows = []
    for name, m in table1.items():
        rows.append(
            {
                "Model / Pipeline Stage": name,
                "Accuracy": round(m.get("accuracy", 0), 4),
                "Precision": round(m.get("precision", 0), 4),
                "Recall": round(m.get("recall", 0), 4),
                "F1-Score": round(m.get("f1", 0), 4),
                "AUC-ROC": round(m.get("auc_roc", 0), 4),
                "Val_Accuracy": round(m.get("val_accuracy", 0), 4),
                "Val_Precision": round(m.get("val_precision", 0), 4),
                "Val_Recall": round(m.get("val_recall", 0), 4),
                "Val_F1-Score": round(m.get("val_f1", 0), 4),
                "Val_AUC-ROC": round(m.get("val_auc_roc", 0), 4),
            }
        )
    df = pd.DataFrame(rows)
    out_csv = MODELS_DIR / "consolidated_results_table.csv"
    df.to_csv(out_csv, index=False)

    print(f"\n" + "=" * 70)
    print(f"  ✓ Training Pipeline Completed Successfully!")
    print(f"    Consolidated table saved to: {out_csv}")
    print(f"    PyTorch Checkpoint & ONNX Model exported to: {MODELS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    run_training_pipeline()
