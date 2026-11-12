"""
IntelliCrash Hardware — Phase 2 Model Training (patched)
=========================================================
Trains Bi-LSTM crash detection model + baselines
on SmartElex Micro ISM330DHCX 6-DoF IMU sensor telemetry.

Run AFTER data_pipeline.py has generated the .npy files.

Run in Colab:
    !pip install torch optuna xgboost lightgbm scikit-learn onnx onnxruntime tqdm scipy
    !python hardware/data_pipeline.py
    !python hardware/train_phase2.py

Output: hardware/models/

Changes vs. previous version
  * load_bilstm(): checkpoint stores hparams; legacy checkpoints still load
  * FocalLoss per-class alpha (in bilstm.py)
  * Weight sweep, thresholds and TOPSIS ranking use VALIDATION data only;
    test metrics are reported once for the selected models
  * Same validation-tuned threshold rule (F-beta, beta=1.5) for every model
  * Block split trims EDGE_TRIM windows at block edges (no 75%-overlap leakage)
  * Raw 6 channels standardised with train statistics
  * Test predictions of every model are saved -> Cell 5 confusion matrix matches tables
  * Block-bootstrap CIs and TOPSIS weight-sensitivity check in Cell 5
"""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede torch import

import numpy as np
import pandas as pd
import sys
import re
import random
import time
import json
from collections import Counter
from pathlib import Path
from tqdm import tqdm

# ── Path Setup ──
HARDWARE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HARDWARE_DIR.parent
sys.path.insert(0, str(HARDWARE_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from bilstm import IntelliCrashBiLSTM, IMUDataset, FocalLoss
    from feature_engineering import NUM_ENGINEERED_FEATURES, FEATURE_NAMES
except ImportError:
    from hardware.bilstm import IntelliCrashBiLSTM, IMUDataset, FocalLoss
    from hardware.feature_engineering import NUM_ENGINEERED_FEATURES, FEATURE_NAMES

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42
WINDOW_SIZE = 200
NUM_RAW_CHANNELS = 6
NUM_FEATURES = NUM_ENGINEERED_FEATURES  # 26
LSTM_INPUT_SIZE = NUM_RAW_CHANNELS + NUM_FEATURES  # 32

DEFAULT_HPARAMS = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "lr": 0.001,
    "batch_size": 64,
    "epochs": 50,
    "weight_decay": 0.01,
    "focal_alpha": 0.75,
    "focal_gamma": 2.0,
    "crash_loss_weight": 0.7,
    "severity_loss_weight": 0.3,
}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Block split: window=200, stride=50 -> windows i and i+3 still overlap.
BLOCK_SIZE = 20
EDGE_TRIM = 3                                   # windows dropped at each block edge
KEPT_PER_BLOCK = BLOCK_SIZE - 2 * EDGE_TRIM     # 14 windows kept per block

THRESH_BETA = 1.5  # F-beta used to pick every model's decision threshold on validation

CSI_G_NORM = 28.0
CSI_DV_NORM = 150.0
CSI_WEIGHTS = {"peak_g": 0.35, "delta_v": 0.30, "jerk": 0.20, "duration": 0.15}

MODELS_DIR = HARDWARE_DIR / "models"
PREDS_PATH = MODELS_DIR / "test_predictions.npz"


# ══════════════════════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════

def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    print(f"[Seed] All random seeds set to {seed} for reproducibility.")


# ══════════════════════════════════════════════════════════════════════════════
#  METRIC / THRESHOLD / PREDICTION-STORE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

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


def _fbeta(y, preds, beta=THRESH_BETA):
    p = precision_score(y, preds, zero_division=0)
    r = recall_score(y, preds, zero_division=0)
    d = beta ** 2 * p + r
    return (1 + beta ** 2) * p * r / d if d > 0 else 0.0


def _tune_threshold(y_val, val_probs, beta=THRESH_BETA):
    """Pick decision threshold on VALIDATION data by maximising F-beta."""
    val_probs = np.asarray(val_probs, dtype=float)
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.10, 0.90, 81):
        s = _fbeta(y_val, (val_probs > t).astype(int), beta)
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t


def _pred_key(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def _save_preds(name, preds):
    store = {}
    if PREDS_PATH.exists():
        with np.load(PREDS_PATH) as d:
            store = {k: d[k] for k in d.files}
    store[_pred_key(name)] = np.asarray(preds).astype(np.int8)
    PREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(PREDS_PATH, **store)


def _load_preds(name):
    if not PREDS_PATH.exists():
        return None
    with np.load(PREDS_PATH) as d:
        k = _pred_key(name)
        return d[k].astype(int) if k in d.files else None


def _evaluate_model(name, y_val, val_probs, y_test, test_probs, thr=None):
    """Tune threshold on val (unless given), report test + val metrics, store test preds."""
    if thr is None:
        thr = _tune_threshold(y_val, val_probs)
    m = _basic_metrics(y_test, test_probs, thr)
    v = _basic_metrics(y_val, val_probs, thr)
    m.update({"val_" + k: x for k, x in v.items()})
    m["threshold"] = float(thr)
    _save_preds(name, (np.asarray(test_probs) > thr).astype(int))
    return m


def _to_rows(results):
    rows = []
    for name, m in results.items():
        rows.append({
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
            "Threshold": round(m.get("threshold", 0.5), 3),
        })
    return pd.DataFrame(rows)


def _print_metrics_table(title, results_dict, extra_header=""):
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    if extra_header:
        print(f"  {extra_header}")
    print(f"{'=' * 100}")
    print(f"\n  {'Model / Pipeline Stage':<48} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} {'F1':>9} {'AUC-ROC':>9} {'Thr':>6}")
    print(f"  {'-'*104}")
    for name, m in results_dict.items():
        auc = m.get("auc_roc", 0)
        auc_str = f"{auc:.4f}" if isinstance(auc, (int, float)) and auc > 0 else "-"
        print(f"  {name:<48} "
              f"{m.get('accuracy', 0):>9.4f} "
              f"{m.get('precision', 0):>9.4f} "
              f"{m.get('recall', 0):>9.4f} "
              f"{m.get('f1', 0):>9.4f} "
              f"{auc_str:>9} "
              f"{m.get('threshold', 0.5):>6.2f}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    data_dir = HARDWARE_DIR / "data"

    print("\n" + "=" * 60)
    print("  SECTION 1: LOADING PREPROCESSED DATA")
    print("=" * 60)

    X = np.load(data_dir / "hw_X.npy")
    y = np.load(data_dir / "hw_y.npy")
    features = np.load(data_dir / "hw_features.npy")
    csi = np.load(data_dir / "hw_csi.npy")

    print(f"  X (raw windows):  {X.shape}")
    print(f"  y (labels):       {y.shape}")
    print(f"  features (26):    {features.shape}")
    print(f"  CSI scores:       {csi.shape}  (range {csi.min():.3f}..{csi.max():.3f})")
    print(f"  CRASH:     {y.sum():,} ({100*y.mean():.1f}%)")
    print(f"  NON_CRASH: {(1-y).sum():,} ({100*(1-y.mean()):.1f}%)")
    if csi.min() < 0 or csi.max() > 1:
        print("  [WARN] CSI is outside [0,1]; the severity head (Sigmoid) and fusion assume [0,1].")

    return X, y, features, csi


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: BLOCK-STRATIFIED SPLIT WITH EDGE TRIM + RAW-CHANNEL SCALING
# ══════════════════════════════════════════════════════════════════════════════

def _block_indices(blocks, block_size=BLOCK_SIZE, trim=EDGE_TRIM):
    return np.concatenate([np.arange(b * block_size + trim, (b + 1) * block_size - trim) for b in blocks])


def split_data(X, y, features, csi, block_size=BLOCK_SIZE):
    print("\n" + "=" * 60)
    print("  SECTION 2: BLOCK-STRATIFIED TRAIN / VAL / TEST SPLIT")
    print("=" * 60)

    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import RobustScaler

    N = len(y)
    num_blocks = N // block_size
    block_y = np.array([y[i * block_size:(i + 1) * block_size].mean() > 0.1 for i in range(num_blocks)])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_b, temp_b = next(sss.split(np.zeros(num_blocks), block_y))

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=SEED)
    val_sub, test_sub = next(sss_val.split(np.zeros(len(temp_b)), block_y[temp_b]))
    val_b = temp_b[val_sub]
    test_b = temp_b[test_sub]

    # Trim EDGE_TRIM windows at both ends of each block so windows in different
    # splits never share raw samples (window 200, stride 50 -> 3-window overlap reach).
    train_idx = _block_indices(train_b, block_size)
    val_idx = _block_indices(val_b, block_size)
    test_idx = _block_indices(test_b, block_size)

    dropped = N - (len(train_idx) + len(val_idx) + len(test_idx))
    print(f"  Edge trim = {EDGE_TRIM} windows/block edge; dropped {dropped} windows "
          f"(incl. {N - num_blocks * block_size} remainder) to prevent overlap leakage.")

    splits = {
        name: {"X": X[idx], "y": y[idx], "features": features[idx], "csi": csi[idx]}
        for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx))
    }

    # Engineered features: RobustScaler fit on train
    scaler = RobustScaler()
    splits["train"]["features_scaled"] = scaler.fit_transform(splits["train"]["features"])
    splits["val"]["features_scaled"] = scaler.transform(splits["val"]["features"])
    splits["test"]["features_scaled"] = scaler.transform(splits["test"]["features"])

    # Raw 6 channels: per-channel standardisation fit on train
    raw_mean = splits["train"]["X"].mean(axis=(0, 1))
    raw_std = splits["train"]["X"].std(axis=(0, 1)) + 1e-8
    for s in splits.values():
        s["X"] = ((s["X"] - raw_mean) / raw_std).astype(np.float32)

    scaler_info = {
        "center": scaler.center_.tolist(),
        "scale": scaler.scale_.tolist(),
        "num_features": len(scaler.center_),
        "raw_mean": raw_mean.tolist(),
        "raw_std": raw_std.tolist(),
    }
    scaler_path = MODELS_DIR / "scaler_params.json"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "w") as f:
        json.dump(scaler_info, f, indent=2)
    print(f"  [OK] Feature scaler + raw-channel stats (train only) saved: {scaler_path}")

    print(f"\n  {'Split':<8} {'Total':>8} {'CRASH':>8} {'NON_CRASH':>10} {'Crash%':>8}")
    print(f"  {'-'*44}")
    for name, s in splits.items():
        n = len(s["y"])
        c = s["y"].sum()
        pct = 100 * c / n if n > 0 else 0
        print(f"  {name:<8} {n:>8,} {int(c):>8,} {int(n - c):>10,} {pct:>7.1f}%")
    print(f"  {'TOTAL':<8} {sum(len(s['y']) for s in splits.values()):>8,}")
    print(f"\n  Seed: {SEED} | Block-stratified, {KEPT_PER_BLOCK} windows kept per block")

    return splits


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: BUILD LSTM TENSORS
# ══════════════════════════════════════════════════════════════════════════════

def build_lstm_input(X_raw, features):
    """(N,200,6) raw + (N,26) features broadcast -> (N,200,32) float32."""
    seq_len = X_raw.shape[1]
    features_broadcast = np.repeat(features[:, np.newaxis, :], seq_len, axis=1)
    return np.concatenate([X_raw, features_broadcast], axis=2).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT LOADING / PREDICTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_bilstm(ckpt_path, device):
    """Load a Bi-LSTM checkpoint (new dict format or legacy raw state_dict)."""
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd, hp = ckpt["state_dict"], ckpt["hparams"]
    else:  # legacy: infer architecture from the weights
        sd = ckpt
        layers = {k.split(".")[1].split("_")[-1] for k in sd
                  if k.startswith("lstm.weight_ih_l") and "reverse" not in k}
        hp = {"hidden_size": sd["lstm.weight_hh_l0"].shape[1],
              "num_layers": len(layers),
              "dropout": DEFAULT_HPARAMS["dropout"]}
    m = IntelliCrashBiLSTM(input_size=LSTM_INPUT_SIZE, hidden_size=hp["hidden_size"],
                           num_layers=hp["num_layers"], dropout=hp["dropout"]).to(device)
    m.load_state_dict(sd)
    return m.eval()


def predict_bilstm(model, splits, split, device):
    feat = splits[split].get("features_scaled", splits[split]["features"])
    X = build_lstm_input(splits[split]["X"], feat)
    ds = IMUDataset(X, splits[split]["y"], splits[split]["csi"])
    loader = DataLoader(ds, batch_size=64)
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            p, _ = model(batch["features"].to(device))
            out.extend(p.cpu().numpy().flatten())
    return np.array(out)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: BI-LSTM TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_bilstm(splits, hparams, device, tag="default"):
    """Returns: model, history, val_metrics, test_metrics, test_probs"""
    print(f"\n  Training Bi-LSTM [{tag}]...")
    print(f"  Hyperparameters: {json.dumps(hparams, indent=2)}")

    feat_train = splits["train"].get("features_scaled", splits["train"]["features"])
    feat_val = splits["val"].get("features_scaled", splits["val"]["features"])
    feat_test = splits["test"].get("features_scaled", splits["test"]["features"])

    X_train = build_lstm_input(splits["train"]["X"], feat_train)
    X_val = build_lstm_input(splits["val"]["X"], feat_val)
    X_test = build_lstm_input(splits["test"]["X"], feat_test)

    train_ds = IMUDataset(X_train, splits["train"]["y"], splits["train"]["csi"])
    val_ds = IMUDataset(X_val, splits["val"]["y"], splits["val"]["csi"])
    test_ds = IMUDataset(X_test, splits["test"]["y"], splits["test"]["csi"])

    set_all_seeds(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=hparams["batch_size"], shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=hparams["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=hparams["batch_size"])

    model = IntelliCrashBiLSTM(
        input_size=LSTM_INPUT_SIZE,
        hidden_size=hparams["hidden_size"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"],
    ).to(device)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion_crash = FocalLoss(alpha=hparams["focal_alpha"], gamma=hparams["focal_gamma"])
    criterion_sev = nn.MSELoss()
    w_crash = hparams["crash_loss_weight"]
    w_sev = hparams["severity_loss_weight"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hparams["epochs"])

    best_val_loss = float("inf")
    best_state = None
    patience, patience_counter = 10, 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(hparams["epochs"]):
        model.train()
        train_loss_sum, train_correct = 0.0, 0
        for batch in train_loader:
            X = batch["features"].to(device)
            y_c = batch["crash"].to(device)
            y_s = batch["severity"].to(device)

            optimizer.zero_grad()
            pred_crash, pred_sev = model(X)
            loss = w_crash * criterion_crash(pred_crash, y_c) + w_sev * criterion_sev(pred_sev, y_s)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * X.size(0)
            train_correct += ((pred_crash > 0.5) == y_c).sum().item()
        scheduler.step()

        train_loss = train_loss_sum / len(train_ds)
        train_acc = train_correct / len(train_ds)

        model.eval()
        val_loss_sum, val_correct = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                X = batch["features"].to(device)
                y_c = batch["crash"].to(device)
                y_s = batch["severity"].to(device)
                pred_crash, pred_sev = model(X)
                loss = w_crash * criterion_crash(pred_crash, y_c) + w_sev * criterion_sev(pred_sev, y_s)
                val_loss_sum += loss.item() * X.size(0)
                val_correct += ((pred_crash > 0.5) == y_c).sum().item()

        val_loss = val_loss_sum / len(val_ds)
        val_acc = val_correct / len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        saved_marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            saved_marker = " ★ BEST"
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or saved_marker:
            print(f"  Epoch {epoch+1:3d}/{hparams['epochs']} | "
                  f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
                  f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}{saved_marker}")

        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    if best_state is None:  # e.g. NaN validation loss
        print("  [WARN] No finite validation loss; using last-epoch weights.")
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    beta = 1.5

    def _collect(loader):
        probs, true = [], []
        with torch.no_grad():
            for batch in loader:
                pc, _ = model(batch["features"].to(device))
                probs.extend(pc.cpu().numpy().flatten())
                true.extend(batch["crash"].numpy().flatten())
        return np.array(probs), np.array(true)

    def _metrics(true, probs):
        preds = (probs > 0.5).astype(int)
        p = precision_score(true, preds, zero_division=0)
        r = recall_score(true, preds, zero_division=0)
        fb = ((1 + beta**2) * p * r) / ((beta**2 * p) + r) if ((beta**2 * p) + r) > 0 else 0.0
        return {
            "accuracy": accuracy_score(true, preds),
            "precision": p, "recall": r,
            "f1": f1_score(true, preds, zero_division=0),
            "f_beta": fb,
            "auc_roc": roc_auc_score(true, probs) if len(np.unique(true)) > 1 else 0.0,
        }

    val_probs, val_true = _collect(val_loader)
    test_probs, test_true = _collect(test_loader)
    return model, history, _metrics(val_true, val_probs), _metrics(test_true, test_probs), test_probs


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: OPTUNA
# ══════════════════════════════════════════════════════════════════════════════

def run_optuna_search(splits, device, n_trials=15):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("\n  [WARN] Optuna not installed. Skipping HP search.  pip install optuna")
        return None

    print("\n" + "=" * 60)
    print("  SECTION 5: HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    print(f"  Running {n_trials} trials (Architecture + Focal Loss + Loss Weights)...")
    print("=" * 60)

    def objective(trial):
        crash_weight = trial.suggest_float("crash_loss_weight", 0.65, 0.85, step=0.05)
        hparams = {
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "epochs": 20,
            "weight_decay": 0.01,
            "focal_alpha": trial.suggest_float("focal_alpha", 0.50, 0.90, step=0.05),
            "focal_gamma": trial.suggest_float("focal_gamma", 1.5, 3.0, step=0.5),
            "crash_loss_weight": crash_weight,
            "severity_loss_weight": round(1.0 - crash_weight, 2),
        }
        set_all_seeds(SEED)
        _, _, val_metrics, _, _ = train_bilstm(splits, hparams, device, tag=f"trial-{trial.number}")
        return val_metrics["f_beta"]  # validation only

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best Trial: #{study.best_trial.number}")
    print(f"  Best Validation F_beta: {study.best_trial.value:.4f}")
    print(f"  Best Params: {json.dumps(study.best_trial.params, indent=4)}")

    best_hparams = DEFAULT_HPARAMS.copy()
    best_hparams.update(study.best_trial.params)
    best_hparams["severity_loss_weight"] = round(1.0 - best_hparams["crash_loss_weight"], 2)
    best_hparams["epochs"] = 50
    return best_hparams


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

def train_xgboost(splits):
    print("\n" + "=" * 60)
    print("  SECTION 6: XGBOOST BASELINES (NORMAL & FINE-TUNED)")
    print("=" * 60)

    try:
        import xgboost as xgb
    except ImportError:
        print("  [WARN] xgboost not installed. Skipping.")
        return None, None, None, None

    X_train, y_train = splits["train"]["features"], splits["train"]["y"]
    X_val, y_val = splits["val"]["features"], splits["val"]["y"]
    X_test, y_test = splits["test"]["features"], splits["test"]["y"]
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    print("  Training XGBoost (Normal / Default)...")
    model_normal = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        objective="binary:logistic", eval_metric="logloss", random_state=SEED,
    )
    model_normal.fit(X_train, y_train, verbose=False)
    m_normal = _evaluate_model("XGBoost (Normal)", y_val, model_normal.predict_proba(X_val)[:, 1],
                               y_test, model_normal.predict_proba(X_test)[:, 1])
    print(f"    XGBoost Normal     Acc={m_normal['accuracy']:.4f} | Prec={m_normal['precision']:.4f} | Rec={m_normal['recall']:.4f} | F1={m_normal['f1']:.4f} | AUC={m_normal['auc_roc']:.4f}")

    print("  Training XGBoost (Fine-Tuned)...")
    model_tuned = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss", random_state=SEED,
    )
    model_tuned.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "hw_xgboost.json"
    model_tuned.save_model(str(model_path))
    print(f"    Saved: {model_path}")

    probs_tuned = model_tuned.predict_proba(X_test)[:, 1]
    val_probs_tuned = model_tuned.predict_proba(X_val)[:, 1]
    m_tuned = _evaluate_model("XGBoost (Fine-Tuned)", y_val, val_probs_tuned, y_test, probs_tuned)
    print(f"    XGBoost Fine-Tuned Acc={m_tuned['accuracy']:.4f} | Prec={m_tuned['precision']:.4f} | Rec={m_tuned['recall']:.4f} | F1={m_tuned['f1']:.4f} | AUC={m_tuned['auc_roc']:.4f}")

    return m_normal, m_tuned, probs_tuned, val_probs_tuned


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6B: CLASSICAL ML
# ══════════════════════════════════════════════════════════════════════════════

def train_classical_baselines(splits):
    print("\n" + "=" * 60)
    print("  SECTION 6B: CLASSICAL ML BASELINES (RF, SVM, LOGISTIC REGRESSION)")
    print("=" * 60)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression

    X_train = splits["train"].get("features_scaled", splits["train"]["features"])
    X_val = splits["val"].get("features_scaled", splits["val"]["features"])
    X_test = splits["test"].get("features_scaled", splits["test"]["features"])
    y_train, y_val, y_test = splits["train"]["y"], splits["val"]["y"], splits["test"]["y"]

    baselines = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=SEED, n_jobs=-1),
        "SVM (RBF Kernel)": SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    }

    out = {}
    for name, clf in baselines.items():
        print(f"  Training {name}...")
        clf.fit(X_train, y_train)
        m = _evaluate_model(name, y_val, clf.predict_proba(X_val)[:, 1], y_test, clf.predict_proba(X_test)[:, 1])
        out[name] = m
        print(f"    {name:<20} Acc={m['accuracy']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | F1={m['f1']:.4f} | AUC={m['auc_roc']:.4f}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6C: DEEP BASELINES
# ══════════════════════════════════════════════════════════════════════════════

class Baseline1DCNN(nn.Module):
    def __init__(self, in_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x.permute(0, 2, 1))


class VanillaLSTM(nn.Module):
    def __init__(self, in_channels=32, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(in_channels, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMAutoencoder(nn.Module):
    def __init__(self, in_channels=32, hidden_size=32):
        super().__init__()
        self.encoder = nn.LSTM(in_channels, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, in_channels, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        latent = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        rec, _ = self.decoder(latent)
        return rec


def _fit_binary(model, loader, device, epochs=15):
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    crit = nn.BCELoss()
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            loss = crit(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def train_deep_baselines(splits, device):
    print("\n" + "=" * 60)
    print("  SECTION 6C: DEEP LEARNING BASELINES (1D-CNN, VANILLA LSTM, AUTOENCODER)")
    print("=" * 60)

    def _inp(split):
        f = splits[split].get("features_scaled", splits[split]["features"])
        return build_lstm_input(splits[split]["X"], f)

    X_train, X_val, X_test = _inp("train"), _inp("val"), _inp("test")
    y_train, y_val, y_test = splits["train"]["y"], splits["val"]["y"], splits["test"]["y"]

    train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1)
    val_tensor = torch.from_numpy(X_val).float().to(device)
    test_tensor = torch.from_numpy(X_test).float().to(device)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(train_tensor, y_train_tensor), batch_size=64, shuffle=True, generator=g)

    def _predict(model, tensor):
        with torch.no_grad():
            return model(tensor).cpu().numpy().flatten()

    def _report(name, m):
        print(f"    {name:<24} Acc={m['accuracy']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | F1={m['f1']:.4f} | AUC={m['auc_roc']:.4f}")

    results = {}

    print("  Training 1D-CNN Baseline (15 epochs)...")
    cnn = _fit_binary(Baseline1DCNN(32).to(device), train_loader, device)
    results["1D CNN"] = _evaluate_model("1D CNN", y_val, _predict(cnn, val_tensor), y_test, _predict(cnn, test_tensor))
    _report("1D CNN", results["1D CNN"])

    print("  Training Vanilla LSTM Baseline (15 epochs)...")
    vl = _fit_binary(VanillaLSTM(32, 64).to(device), train_loader, device)
    results["Vanilla LSTM"] = _evaluate_model("Vanilla LSTM", y_val, _predict(vl, val_tensor), y_test, _predict(vl, test_tensor))
    _report("Vanilla LSTM", results["Vanilla LSTM"])

    print("  Training Unsupervised LSTM Autoencoder on Normal Driving (15 epochs)...")
    normal_train = train_tensor[torch.from_numpy(y_train == 0)]
    ae_loader = DataLoader(TensorDataset(normal_train), batch_size=64, shuffle=True, generator=g)
    ae = LSTMAutoencoder(32, 32).to(device)
    opt_ae = torch.optim.Adam(ae.parameters(), lr=0.002, weight_decay=1e-4)
    mse_crit = nn.MSELoss()
    for _ in range(15):
        ae.train()
        for (bx,) in ae_loader:
            bx = bx.to(device)
            loss = mse_crit(ae(bx), bx)
            opt_ae.zero_grad()
            loss.backward()
            opt_ae.step()
    ae.eval()
    with torch.no_grad():
        val_mse = torch.mean((val_tensor - ae(val_tensor)) ** 2, dim=(1, 2)).cpu().numpy()
        test_mse = torch.mean((test_tensor - ae(test_tensor)) ** 2, dim=(1, 2)).cpu().numpy()
    # Monotone map of reconstruction error to (0,1); scale taken from validation only.
    ref = np.percentile(val_mse, 90) + 1e-12
    val_score = val_mse / (val_mse + ref)
    test_score = test_mse / (test_mse + ref)
    results["Unsup. LSTM Autoencoder"] = _evaluate_model("Unsup. LSTM Autoencoder", y_val, val_score, y_test, test_score)
    _report("Unsup. LSTM Autoencoder", results["Unsup. LSTM Autoencoder"])

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6D: LIGHTGBM
# ══════════════════════════════════════════════════════════════════════════════

def train_lightgbm(splits):
    try:
        import lightgbm as lgb
    except ImportError:
        print("  [INFO] lightgbm not installed. Skipping LightGBM.")
        return None, None, None

    print("\n" + "=" * 60)
    print("  SECTION 6D: LIGHTGBM TABULAR BASELINE")
    print("=" * 60)

    X_train, y_train = splits["train"]["features"], splits["train"]["y"]
    X_val, y_val = splits["val"]["features"], splits["val"]["y"]
    X_test, y_test = splits["test"]["features"], splits["test"]["y"]

    model = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=SEED, verbose=-1)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]
    m = _evaluate_model("LightGBM", y_val, val_probs, y_test, probs)
    print(f"    LightGBM           Acc={m['accuracy']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | F1={m['f1']:.4f} | AUC={m['auc_roc']:.4f}")
    return m, probs, val_probs


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7A: CSI ONLY
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_csi_only(splits):
    """CSI-only baseline; threshold tuned on validation."""
    y_val, csi_val = splits["val"]["y"], splits["val"]["csi"]
    y_test, csi_test = splits["test"]["y"], splits["test"]["csi"]
    m = _evaluate_model("CSI Only (Physics Path)", y_val, csi_val, y_test, csi_test)
    thr = m["threshold"]
    print(f"    CSI Only (τ={thr:.2f}) Acc={m['accuracy']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | F1={m['f1']:.4f} | AUC={m['auc_roc']:.4f}")
    return m, thr


# ══════════════════════════════════════════════════════════════════════════════
#  TOPSIS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _topsis_scores(data, weights, impacts):
    data = np.asarray(data, dtype=float)
    rss = np.sqrt((data ** 2).sum(axis=0))
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


def _run_topsis_internal(input_df, weights, impacts, output_path=None):
    """TOPSIS via external `topsis` package if available, else built-in fallback."""
    try:
        from topsis import topsis
        topsis(input_df, weights, impacts, str(output_path) if output_path else "")
        if output_path and Path(output_path).exists():
            res = pd.read_csv(output_path)
            if "Topsis Score" in res.columns and "Rank" in res.columns:
                return res
    except Exception:
        pass

    df = input_df.copy()
    data = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    scores = _topsis_scores(data, weights, impacts)
    df["Topsis Score"] = scores
    df["Rank"] = pd.Series(scores).rank(ascending=False, method="dense").astype(int)
    ranked = df.sort_values("Rank").reset_index(drop=True)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(output_path, index=False)
    return ranked


def _compute_fpr(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0][0], cm[0][1]
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7B: BI-MODAL FUSION (weights/threshold chosen on VALIDATION)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_bimodal_fusion(splits, model, device, lstm_test_probs, lstm_val_probs):
    print("\n" + "=" * 60)
    print("  SECTION 7B: BI-MODAL FUSION WEIGHT OPTIMIZATION (VALIDATION SET)")
    print("  Mini-TOPSIS: Recall(+), FPR(-), F1(+)")
    print("=" * 60)

    y_val, csi_val = splits["val"]["y"], splits["val"]["csi"]
    y_test, csi_test = splits["test"]["y"], splits["test"]["csi"]

    weight_configs = [
        {"name": "1.0 : 0.0", "w_lstm": 1.0, "w_csi": 0.0},
        {"name": "0.9 : 0.1", "w_lstm": 0.9, "w_csi": 0.1},
        {"name": "0.8 : 0.2", "w_lstm": 0.8, "w_csi": 0.2},
        {"name": "0.7 : 0.3", "w_lstm": 0.7, "w_csi": 0.3},
        {"name": "0.6 : 0.4", "w_lstm": 0.6, "w_csi": 0.4},
        {"name": "0.5 : 0.5", "w_lstm": 0.5, "w_csi": 0.5},
        {"name": "0.4 : 0.6", "w_lstm": 0.4, "w_csi": 0.6},
        {"name": "0.0 : 1.0", "w_lstm": 0.0, "w_csi": 1.0},
    ]

    bimodal_results, rows = {}, []
    for cfg in weight_configs:
        w, wc = cfg["w_lstm"], cfg["w_csi"]
        p_val = w * lstm_val_probs + wc * csi_val
        p_test = w * lstm_test_probs + wc * csi_test
        tau = _tune_threshold(y_val, p_val)

        vm = _basic_metrics(y_val, p_val, tau)
        val_preds = (p_val > tau).astype(int)
        tm = _basic_metrics(y_test, p_test, tau)
        bimodal_results[cfg["name"]] = {**tm, "w_lstm": w, "w_csi": wc, "tau": tau}

        rows.append({
            "Fusion Weights (ML:CSI)": cfg["name"],
            "Recall (%)": round(vm["recall"] * 100, 2),
            "FPR (%)": round(_compute_fpr(y_val, val_preds) * 100, 2),
            "F1-Score (%)": round(vm["f1"] * 100, 2),
            "Threshold": round(tau, 2),
        })

    weight_opt_df = pd.DataFrame(rows)
    topsis_input = weight_opt_df[["Fusion Weights (ML:CSI)", "Recall (%)", "FPR (%)", "F1-Score (%)"]].copy()
    topsis_result = _run_topsis_internal(topsis_input, [0.40, 0.30, 0.30], ['+', '-', '+'],
                                         MODELS_DIR / "bimodal_weight_topsis.csv")
    topsis_result = topsis_result.merge(weight_opt_df[["Fusion Weights (ML:CSI)", "Threshold"]],
                                        on="Fusion Weights (ML:CSI)", how="left")

    # 1.0:0.0 (pure Bi-LSTM) and 0.0:1.0 (pure CSI) are boundary baselines, not fusions
    fusion_only = topsis_result[~topsis_result["Fusion Weights (ML:CSI)"].isin(["1.0 : 0.0", "0.0 : 1.0"])]
    best_row = fusion_only.loc[fusion_only["Rank"] == fusion_only["Rank"].min()].iloc[0]
    best_name = best_row["Fusion Weights (ML:CSI)"]
    best_cfg = bimodal_results[best_name]

    parts = str(best_name).split(":")
    with open(MODELS_DIR / "optimal_fusion_weights.json", "w") as f:
        json.dump({
            "w_ml": float(parts[0].strip()),
            "w_csi": float(parts[1].strip()),
            "threshold": best_cfg["tau"],
            "topsis_score": float(best_row["Topsis Score"]),
            "best_weights": best_name,
            "selected_on": "validation",
        }, f, indent=2)

    return bimodal_results, best_name, best_cfg, topsis_result, lstm_val_probs


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7C: TRI-MODAL FUSION (weights/threshold chosen on VALIDATION)
# ══════════════════════════════════════════════════════════════════════════════

def _optimize_trimodal_weights(lstm_val_probs, tab_val_probs, csi_val, y_val):
    """Grid search (w1,w2,w3,tau) on validation maximising F-beta (same rule as other models)."""
    best_score = -1.0
    best = (0.25, 0.55, 0.20, 0.45)
    cands = []
    for w1 in np.linspace(0.05, 0.60, 12):
        for w2 in np.linspace(0.10, 0.75, 14):
            w3 = round(1.0 - w1 - w2, 2)
            if w3 >= 0.05:
                cands.append((round(w1, 2), round(w2, 2), w3))
    for w1, w2, w3 in cands:
        p_val = w1 * lstm_val_probs + w2 * tab_val_probs + w3 * csi_val
        for tau in np.linspace(0.20, 0.70, 26):
            score = _fbeta(y_val, (p_val > tau).astype(int))
            if score > best_score:
                best_score, best = score, (w1, w2, w3, float(tau))
    return best


def evaluate_trimodal_fusion(splits, model, device, lstm_test_probs, lstm_val_probs,
                             xgb_probs, xgb_val_probs, lgb_probs=None, lgb_val_probs=None):
    print("\n" + "=" * 60)
    print("  SECTION 7C: TRI-MODAL FUSION EVALUATION")
    print("=" * 60)

    y_val, csi_val = splits["val"]["y"], splits["val"]["csi"]
    y_test, csi_test = splits["test"]["y"], splits["test"]["csi"]

    variants = []
    if xgb_probs is not None and xgb_val_probs is not None:
        variants.append(("Tri-Modal (BiLSTM + CSI + XGBoost)", xgb_probs, xgb_val_probs, "xgb"))
    if lgb_probs is not None and lgb_val_probs is not None:
        variants.append(("Tri-Modal (BiLSTM + CSI + LightGBM)", lgb_probs, lgb_val_probs, "lgb"))

    results, params = {}, {}
    for name, tab_probs, tab_val_probs, key in variants:
        print(f"\n  Optimizing: {name}...")
        w1, w2, w3, tau = _optimize_trimodal_weights(lstm_val_probs, tab_val_probs, csi_val, y_val)
        print(f"    w_LSTM={w1:.2f}, w_{key.upper()}={w2:.2f}, w_CSI={w3:.2f} | τ={tau:.2f}")

        p_val = w1 * lstm_val_probs + w2 * tab_val_probs + w3 * csi_val
        p_test = w1 * lstm_test_probs + w2 * tab_probs + w3 * csi_test
        m = _evaluate_model(name, y_val, p_val, y_test, p_test, thr=tau)
        results[name] = m
        params[name] = {"w_lstm": w1, f"w_{key}": w2, "w_csi": w3, "tau_threshold": round(tau, 3)}
        print(f"    {name:<45} Acc={m['accuracy']:.4f} | Prec={m['precision']:.4f} | Rec={m['recall']:.4f} | F1={m['f1']:.4f} | AUC={m['auc_roc']:.4f}")

    with open(MODELS_DIR / "trimodal_params.json", "w") as f:
        json.dump(params, f, indent=2)
    return results, params


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8: ONNX EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_onnx(model, device):
    print("\n" + "=" * 60)
    print("  SECTION 8: ONNX EXPORT")
    print("=" * 60)

    out_path = MODELS_DIR / "hw_bilstm.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.randn(1, WINDOW_SIZE, LSTM_INPUT_SIZE).to(device)

    try:
        try:
            import onnxscript  # noqa: F401
        except ImportError:
            import subprocess
            print("  Installing onnxscript for PyTorch ONNX exporter...")
            subprocess.run([sys.executable, "-m", "pip", "install", "onnxscript", "onnx"], check=False)

        torch.onnx.export(
            model, dummy, str(out_path),
            export_params=True, opset_version=14, do_constant_folding=True,
            input_names=["input"], output_names=["crash_prob", "severity"],
            dynamic_axes={"input": {0: "batch_size"}, "crash_prob": {0: "batch_size"}, "severity": {0: "batch_size"}},
        )
        print(f"  Exported: {out_path}  ({out_path.stat().st_size / 1024:.1f} KB)")

        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(out_path))
            ort_crash, ort_sev = session.run(None, {session.get_inputs()[0].name: dummy.cpu().numpy()})
            print(f"  ONNX verification: crash_prob={ort_crash[0][0]:.4f}, severity={ort_sev[0][0]:.4f}")
            print("  ✓ ONNX model verified successfully!")
        except Exception as e:
            print(f"  [WARN] onnxruntime verification notice: {e}")
    except Exception as e:
        print(f"  [WARN] ONNX export encountered an issue: {e}")
        print("  Note: PyTorch checkpoint (hw_best_bilstm.pth) is saved in hardware/models/")


# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CACHE
# ══════════════════════════════════════════════════════════════════════════════

_GLOBAL_SPLITS = None
_GLOBAL_BASELINES = None
_GLOBAL_BILSTM = None
_GLOBAL_TRIMODAL = None


def get_or_load_splits(splits=None):
    global _GLOBAL_SPLITS
    if splits is not None:
        _GLOBAL_SPLITS = splits
        return _GLOBAL_SPLITS
    if _GLOBAL_SPLITS is not None:
        return _GLOBAL_SPLITS
    set_all_seeds(SEED)
    X, y, features, csi = load_data()
    _GLOBAL_SPLITS = split_data(X, y, features, csi)
    return _GLOBAL_SPLITS


# ══════════════════════════════════════════════════════════════════════════════
#  CELL 1: BASELINES
# ══════════════════════════════════════════════════════════════════════════════

def run_cell_1_baselines(splits=None):
    global _GLOBAL_BASELINES
    print("\n" + "=" * 90)
    print("  CELL 1: BASELINE MODELS (PHYSICAL SMARTELEX ISM330DHCX 6-DOF IMU TESTBED)")
    print("=" * 90)

    set_all_seeds(SEED)
    splits = get_or_load_splits(splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Execution Device: {device}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    classical = train_classical_baselines(splits)
    deep = train_deep_baselines(splits, device)
    xgb_normal_m, xgb_tuned_m, xgb_probs, xgb_val_probs = train_xgboost(splits)
    lgb_metrics, lgb_probs, lgb_val_probs = train_lightgbm(splits)

    table1 = {}
    table1.update(classical)
    table1.update(deep)
    if xgb_normal_m:
        table1["XGBoost (Normal)"] = xgb_normal_m
    if xgb_tuned_m:
        table1["XGBoost (Fine-Tuned)"] = xgb_tuned_m
    if lgb_metrics:
        table1["LightGBM"] = lgb_metrics

    _print_metrics_table("TABLE 1: BASELINE MODELS COMPARISON (TEST SET)", table1,
                         "Thresholds tuned on validation (F-beta=1.5) for every model")

    table1_df = _to_rows(table1)
    out_csv = MODELS_DIR / "table1_baselines.csv"
    table1_df.to_csv(out_csv, index=False)
    print(f"\n  [OK] Table 1 saved: {out_csv}")

    _GLOBAL_BASELINES = {
        "metrics": table1,
        "xgb_probs": xgb_probs, "xgb_val_probs": xgb_val_probs,
        "lgb_probs": lgb_probs, "lgb_val_probs": lgb_val_probs,
    }
    np.savez(
        MODELS_DIR / "baseline_cache.npz",
        xgb_probs=xgb_probs if xgb_probs is not None else np.array([]),
        xgb_val_probs=xgb_val_probs if xgb_val_probs is not None else np.array([]),
        lgb_probs=lgb_probs if lgb_probs is not None else np.array([]),
        lgb_val_probs=lgb_val_probs if lgb_val_probs is not None else np.array([]),
    )
    print("  ✓ Cell 1 Complete! Next → Cell 2: run_cell_2_weight_optimization()")
    return table1_df


# ══════════════════════════════════════════════════════════════════════════════
#  CELL 2: BI-LSTM + FUSION WEIGHT OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

def run_cell_2_weight_optimization(splits=None, run_optuna=True):
    global _GLOBAL_BILSTM
    print("\n" + "=" * 90)
    print("  CELL 2: WEIGHT OPTIMIZATION FOR HYBRID FUSION GATE")
    print("=" * 90)

    set_all_seeds(SEED)
    splits = get_or_load_splits(splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  [Stage 2A] Initial Bi-LSTM Training (Default Hyperparameters)...")
    set_all_seeds(SEED)
    model_default, _, val_m_default, m_default, probs_default = train_bilstm(
        splits, DEFAULT_HPARAMS, device, tag="default-hparams")

    best_hparams = None
    if run_optuna:
        print("\n  [Stage 2B] Running Optuna Bayesian Hyperparameter Optimization (15 trials)...")
        best_hparams = run_optuna_search(splits, device, n_trials=15)

    used_hparams = DEFAULT_HPARAMS
    if best_hparams is not None:
        print("\n  [Stage 2C] Final Bi-LSTM Training (Optimized Hyperparameters)...")
        set_all_seeds(SEED)
        model_best, _, val_m_best, m_best, probs_best = train_bilstm(
            splits, best_hparams, device, tag="optimized-hparams")
        fb_best = val_m_best.get("f_beta", 0)
        fb_def = val_m_default.get("f_beta", 0)
        if fb_best >= fb_def:
            final_model, final_metrics, final_probs, final_tag = model_best, m_best, probs_best, "optimized"
            used_hparams = best_hparams
            print(f"  [Selection] Optimized model selected (Val F_beta: {fb_best:.4f} vs {fb_def:.4f})")
        else:
            final_model, final_metrics, final_probs, final_tag = model_default, m_default, probs_default, "default"
            print(f"  [Selection] Default model selected (Val F_beta: {fb_def:.4f} vs {fb_best:.4f})")
    else:
        final_model, final_metrics, final_probs, final_tag = model_default, m_default, probs_default, "default"

    ckpt_path = MODELS_DIR / "hw_best_bilstm.pth"
    torch.save({"state_dict": final_model.state_dict(),
                "hparams": {k: used_hparams[k] for k in ("hidden_size", "num_layers", "dropout")}},
               ckpt_path)
    print(f"  Saved best Bi-LSTM checkpoint: {ckpt_path}  ({final_tag}, "
          f"layers={used_hparams['num_layers']}, hidden={used_hparams['hidden_size']})")

    val_probs = predict_bilstm(final_model, splits, "val", device)

    print("\n  [Stage 2D] Evaluating Hybrid Fusion Gate Weights (Mini-TOPSIS, validation)...")
    (bimodal_results, best_name, best_cfg, topsis_df, lstm_val_probs) = evaluate_bimodal_fusion(
        splits, final_model, device, final_probs, val_probs)

    table2_df = topsis_df.copy()
    out_csv = MODELS_DIR / "table2_bimodal_weight_optimization.csv"
    table2_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 90)
    print("  TABLE 2: HYBRID FUSION GATE WEIGHT OPTIMIZATION (MINI-TOPSIS, VALIDATION METRICS)")
    print("=" * 90)
    print(table2_df.to_string(index=False))
    print(f"\n  [OK] Table 2 saved: {out_csv}")
    print(f"  Selected Optimal Hybrid Weights: {best_name}")

    _GLOBAL_BILSTM = {
        "model": final_model, "metrics": final_metrics,
        "test_probs": final_probs, "val_probs": lstm_val_probs,
        "best_bimodal_name": best_name, "best_bimodal_config": best_cfg, "tag": final_tag,
    }
    np.savez(MODELS_DIR / "bilstm_cache.npz", test_probs=final_probs, val_probs=lstm_val_probs)

    print("  ✓ Cell 2 Complete! Next → Cell 3: run_cell_3_ablation_study()")
    return table2_df, best_name


# ══════════════════════════════════════════════════════════════════════════════
#  helper: make sure Bi-LSTM probabilities are available
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_bilstm(splits, device):
    global _GLOBAL_BILSTM
    if _GLOBAL_BILSTM is not None and "val_probs" in _GLOBAL_BILSTM:
        return
    ckpt_path = MODELS_DIR / "hw_best_bilstm.pth"
    cache = MODELS_DIR / "bilstm_cache.npz"
    if cache.exists():
        d = np.load(cache)
        _GLOBAL_BILSTM = {"test_probs": d["test_probs"], "val_probs": d["val_probs"]}
    elif ckpt_path.exists():
        mdl = load_bilstm(ckpt_path, device)
        _GLOBAL_BILSTM = {"model": mdl,
                          "test_probs": predict_bilstm(mdl, splits, "test", device),
                          "val_probs": predict_bilstm(mdl, splits, "val", device)}
    else:
        print("  [INFO] Bi-LSTM not found. Running Cell 2 first...")
        run_cell_2_weight_optimization(splits, run_optuna=False)


# ══════════════════════════════════════════════════════════════════════════════
#  CELL 3: ABLATION
# ══════════════════════════════════════════════════════════════════════════════

def run_cell_3_ablation_study(splits=None):
    print("\n" + "=" * 90)
    print("  CELL 3: ABLATION STUDY — CRITICAL SYSTEM COMPONENTS")
    print("=" * 90)

    set_all_seeds(SEED)
    splits = get_or_load_splits(splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt_json = MODELS_DIR / "optimal_fusion_weights.json"
    if not opt_json.exists():
        print("  [INFO] optimal_fusion_weights.json missing. Running Cell 2 first...")
        run_cell_2_weight_optimization(splits, run_optuna=False)
    _ensure_bilstm(splits, device)

    y_val, y_test = splits["val"]["y"], splits["test"]["y"]
    csi_val, csi_test = splits["val"]["csi"], splits["test"]["csi"]
    lstm_val, lstm_test = _GLOBAL_BILSTM["val_probs"], _GLOBAL_BILSTM["test_probs"]

    csi_metrics, _ = evaluate_csi_only(splits)

    lstm_metrics = _evaluate_model("Bi-LSTM Only (Deep Temporal)", y_val, lstm_val, y_test, lstm_test)

    with open(opt_json) as f:
        opt = json.load(f)
    best_name = opt["best_weights"]
    w_lstm, w_csi, tau = opt["w_ml"], opt["w_csi"], opt["threshold"]
    fusion_name = f"Hybrid Fusion Gate (Optimized {best_name})"
    fusion_metrics = _evaluate_model(fusion_name, y_val, w_lstm * lstm_val + w_csi * csi_val,
                                     y_test, w_lstm * lstm_test + w_csi * csi_test, thr=tau)

    table3 = {
        "CSI Only (Physics Path)": csi_metrics,
        "Bi-LSTM Only (Deep Temporal)": lstm_metrics,
        fusion_name: fusion_metrics,
    }
    _print_metrics_table("TABLE 3: ABLATION STUDY — SYSTEM COMPONENTS (TEST SET)", table3,
                         "Physics vs. Deep Sequence vs. Dual-Path Late Fusion")

    table3_df = _to_rows(table3)
    out_csv = MODELS_DIR / "table3_ablation_study.csv"
    table3_df.to_csv(out_csv, index=False)
    print(f"\n  [OK] Table 3 saved: {out_csv}")
    print("  ✓ Cell 3 Complete! Next → Cell 4: run_cell_4_trimodal_fusion()")
    return table3_df


# ══════════════════════════════════════════════════════════════════════════════
#  CELL 4: TRI-MODAL
# ══════════════════════════════════════════════════════════════════════════════

def run_cell_4_trimodal_fusion(splits=None):
    global _GLOBAL_TRIMODAL, _GLOBAL_BASELINES
    print("\n" + "=" * 90)
    print("  CELL 4: TRI-MODAL FUSION VARIANTS (TEST SET)")
    print("=" * 90)

    set_all_seeds(SEED)
    splits = get_or_load_splits(splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _GLOBAL_BASELINES is None:
        cache_file = MODELS_DIR / "baseline_cache.npz"
        if cache_file.exists():
            d = np.load(cache_file)

            def _get(k):
                return d[k] if k in d.files and len(d[k]) > 0 else None
            _GLOBAL_BASELINES = {k: _get(k) for k in ("xgb_probs", "xgb_val_probs", "lgb_probs", "lgb_val_probs")}
        else:
            print("  [INFO] Baseline predictions not found. Running Cell 1 first...")
            run_cell_1_baselines(splits)

    _ensure_bilstm(splits, device)

    results, params = evaluate_trimodal_fusion(
        splits, model=None, device=device,
        lstm_test_probs=_GLOBAL_BILSTM["test_probs"], lstm_val_probs=_GLOBAL_BILSTM["val_probs"],
        xgb_probs=_GLOBAL_BASELINES.get("xgb_probs"), xgb_val_probs=_GLOBAL_BASELINES.get("xgb_val_probs"),
        lgb_probs=_GLOBAL_BASELINES.get("lgb_probs"), lgb_val_probs=_GLOBAL_BASELINES.get("lgb_val_probs"),
    )
    _GLOBAL_TRIMODAL = {"results": results, "params": params}

    _print_metrics_table("TABLE 4: TRI-MODAL FUSION VARIANTS (TEST SET)", results,
                         "3-Way Fusion: Temporal Bi-LSTM + Newtonian CSI + Tabular GBDT")
    table4_df = _to_rows(results)
    out_csv = MODELS_DIR / "table4_trimodal_fusion.csv"
    table4_df.to_csv(out_csv, index=False)
    print(f"\n  [OK] Table 4 saved: {out_csv}")
    print("  ✓ Cell 4 Complete! Next → Cell 5: run_cell_5_consolidated_topsis()")
    return table4_df


# ══════════════════════════════════════════════════════════════════════════════
#  CELL 5: CONSOLIDATED TOPSIS (ranking on VALIDATION metrics)
# ══════════════════════════════════════════════════════════════════════════════

def _read_table(path, runner, splits, force):
    """Reuse a saved table only if it has the validation columns; otherwise recompute."""
    if not force and path.exists():
        df = pd.read_csv(path)
        if "Val_Recall" in df.columns:
            return df
        print(f"  [INFO] {path.name} is from an older run (no validation columns). Recomputing...")
    return runner(splits)


def _block_bootstrap_ci(y, preds, block_len=KEPT_PER_BLOCK, n_boot=1000, seed=SEED):
    y, preds = np.asarray(y), np.asarray(preds)
    nb = len(y) // block_len
    rng = np.random.default_rng(seed)
    off = np.arange(block_len)
    rec, f1 = [], []
    for _ in range(n_boot):
        s = rng.integers(0, nb, nb)
        idx = (s[:, None] * block_len + off).ravel()
        yt, yp = y[idx], preds[idx]
        tp = np.sum((yt == 1) & (yp == 1)); fn = np.sum((yt == 1) & (yp == 0)); fp = np.sum((yt == 0) & (yp == 1))
        rec.append(tp / (tp + fn) if tp + fn else 0.0)
        f1.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return np.percentile(rec, [2.5, 97.5]), np.percentile(f1, [2.5, 97.5]), nb


def run_cell_5_consolidated_topsis(splits=None, force=False):
    print("\n" + "=" * 105)
    print("  CELL 5: CONSOLIDATED ALL-MODELS COMPARISON & TOPSIS DECISION ANALYSIS")
    print("=" * 105)

    splits = get_or_load_splits(splits)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t1 = _read_table(MODELS_DIR / "table1_baselines.csv", run_cell_1_baselines, splits, force)
    t3 = _read_table(MODELS_DIR / "table3_ablation_study.csv", run_cell_3_ablation_study, splits, force)
    t4 = _read_table(MODELS_DIR / "table4_trimodal_fusion.csv", run_cell_4_trimodal_fusion, splits, force)

    cons = pd.concat([t for t in (t1, t3, t4) if t is not None and len(t) > 0], ignore_index=True)
    cons["Model / Pipeline Stage"] = cons["Model / Pipeline Stage"].str.replace("★", "").str.strip()
    cons = cons.drop_duplicates(subset=["Model / Pipeline Stage"]).reset_index(drop=True)

    print("\n  CONSOLIDATED COMPARISON TABLE — TEST SET (thresholds tuned on validation)")
    print(f"  {'Model / Pipeline Stage':<48} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} {'F1':>9} {'AUC-ROC':>9}")
    print(f"  {'-'*100}")
    for _, r in cons.iterrows():
        auc = f"{r['AUC-ROC']:.4f}" if r['AUC-ROC'] > 0 else "-"
        print(f"  {r['Model / Pipeline Stage']:<48} {r['Accuracy']:>9.4f} {r['Precision']:>9.4f} "
              f"{r['Recall']:>9.4f} {r['F1-Score']:>9.4f} {auc:>9}")

    cons_path = MODELS_DIR / "consolidated_results_table.csv"
    cons.to_csv(cons_path, index=False)
    cons.to_csv(MODELS_DIR / "results_comparison_table.csv", index=False)
    print(f"\n  [OK] Consolidated table saved: {cons_path}")

    # ── TOPSIS on VALIDATION metrics ──
    print("\n" + "=" * 105)
    print("  TOPSIS — RANKING ON VALIDATION METRICS (test set is not used for selection)")
    print("  Weights: Recall (0.40), AUC-ROC (0.25), F1 (0.20), Precision (0.10), Accuracy (0.05)")
    print("=" * 105)

    val_cols = ["Val_Accuracy", "Val_Precision", "Val_Recall", "Val_F1-Score", "Val_AUC-ROC"]
    weights = [0.05, 0.10, 0.40, 0.20, 0.25]
    impacts = ['+'] * 5
    topsis_in = cons[["Model / Pipeline Stage"] + val_cols].copy()
    res = _run_topsis_internal(topsis_in, weights, impacts, MODELS_DIR / "topsis_hardware_ranking.csv")
    res = res[["Model / Pipeline Stage", "Topsis Score", "Rank"]].merge(cons, on="Model / Pipeline Stage", how="left")
    res = res.sort_values("Rank").reset_index(drop=True)
    res.to_csv(MODELS_DIR / "topsis_hardware_ranking.csv", index=False)

    print(f"\n  {'Rank':>4}  {'Model / Pipeline Stage':<48} {'ValRec':>7} {'ValAUC':>7} | {'TestAcc':>7} {'TestPrec':>8} {'TestRec':>7} {'TestF1':>7} {'TestAUC':>7} {'Ci':>8}")
    print(f"  {'-'*128}")
    for _, r in res.iterrows():
        print(f"  {int(r['Rank']):>4}  {r['Model / Pipeline Stage']:<48} {r['Val_Recall']:>7.4f} {r['Val_AUC-ROC']:>7.4f} | "
              f"{r['Accuracy']:>7.4f} {r['Precision']:>8.4f} {r['Recall']:>7.4f} {r['F1-Score']:>7.4f} {r['AUC-ROC']:>7.4f} {r['Topsis Score']:>8.4f}")

    best_row = res.loc[res["Rank"] == 1].iloc[0]
    best_name, best_score = best_row["Model / Pipeline Stage"], float(best_row["Topsis Score"])

    print(f"\n{'=' * 90}")
    print(f"  ★ TOPSIS BEST MODEL SELECTED (validation): {best_name}")
    print(f"    TOPSIS Closeness (Ci): {best_score:.4f}   |   Rank #1 among {len(res)} models")
    print(f"{'=' * 90}")

    # ── Weight sensitivity ──
    data = topsis_in[val_cols].to_numpy(dtype=float)
    names = topsis_in["Model / Pipeline Stage"].tolist()
    rng = np.random.default_rng(SEED)
    wins = Counter()
    for _ in range(500):
        w = rng.dirichlet(np.array(weights) * 50)
        wins[names[int(np.argmax(_topsis_scores(data, w, impacts)))]] += 1
    print("\n  TOPSIS weight-sensitivity (500 random perturbations of the weights): share of runs ranked #1")
    for n, c in wins.most_common(5):
        print(f"    {n:<48} {100*c/500:5.1f}%")

    # ── Block-bootstrap CIs on TEST predictions of top-5 ──
    print("\n  Block-bootstrap 95% CIs on TEST set (resampling whole blocks) — top 5 by validation TOPSIS")
    y_test = splits["test"]["y"]
    for _, r in res.head(5).iterrows():
        p = _load_preds(r["Model / Pipeline Stage"])
        if p is None or len(p) != len(y_test):
            print(f"    {r['Model / Pipeline Stage']:<48} (no saved predictions)")
            continue
        (rl, rh), (fl, fh), nb = _block_bootstrap_ci(y_test, p)
        print(f"    {r['Model / Pipeline Stage']:<48} Recall {r['Recall']:.3f} [{rl:.3f}, {rh:.3f}]   F1 {r['F1-Score']:.3f} [{fl:.3f}, {fh:.3f}]  ({nb} test blocks)")
    print("  If intervals overlap, call the models 'comparable', not 'better'.")

    _evaluate_and_export_best_model(best_name, best_score, splits, device)
    return res, best_name


def _evaluate_and_export_best_model(best_model_name, best_score, splits, device):
    print("\n" + "=" * 90)
    print(f"  BEST MODEL DEEP EVALUATION — {best_model_name}")
    print("=" * 90)

    y_test = splits["test"]["y"]
    y_pred = _load_preds(best_model_name)

    lstm_mdl = None
    ckpt_path = MODELS_DIR / "hw_best_bilstm.pth"
    if ckpt_path.exists():
        lstm_mdl = load_bilstm(ckpt_path, device)

    if y_pred is None or len(y_pred) != len(y_test):
        print(f"  [WARN] No saved test predictions for '{best_model_name}'. "
              f"Re-run Cells 1-4 (delete models/*.npz first) to regenerate.")
    else:
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        print(f"\n  Confusion Matrix ({best_model_name}):")
        print(f"                    Predicted")
        print(f"                    NON_CRASH  CRASH")
        print(f"  Actual NON_CRASH  {cm[0][0]:>6}   {cm[0][1]:>6}")
        print(f"  Actual CRASH      {cm[1][0]:>6}   {cm[1][1]:>6}")

        report = classification_report(y_test, y_pred, target_names=["NON_CRASH", "CRASH"], zero_division=0)
        print(f"\n  Classification Report:\n{report}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay

            fig, ax = plt.subplots(figsize=(7, 5.5))
            ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NON_CRASH", "CRASH"]).plot(
                ax=ax, cmap="Blues", values_format="d")
            ax.set_title(f"Confusion Matrix — {best_model_name}\nTOPSIS Score: {best_score:.4f}",
                         fontsize=12, fontweight="bold")
            fig.tight_layout()
            cm_img = MODELS_DIR / "confusion_matrix_best_model.png"
            fig.savefig(str(cm_img), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [OK] Confusion matrix plot saved: {cm_img}")
            try:
                from IPython.display import Image, display
                display(Image(filename=str(cm_img)))
            except Exception:
                pass
        except Exception as e:
            print(f"  [WARN] Could not plot confusion matrix: {e}")

        eval_txt = MODELS_DIR / "best_model_evaluation.txt"
        with open(eval_txt, "w") as f:
            f.write(f"Best Model (TOPSIS Rank #1, selected on validation): {best_model_name}\n")
            f.write(f"TOPSIS Score: {best_score:.4f}\n\nConfusion Matrix:\n")
            f.write("                    Predicted\n                    NON_CRASH  CRASH\n")
            f.write(f"  Actual NON_CRASH  {cm[0][0]:>6}   {cm[0][1]:>6}\n")
            f.write(f"  Actual CRASH      {cm[1][0]:>6}   {cm[1][1]:>6}\n\nClassification Report:\n{report}")
        print(f"  [OK] Evaluation text report saved: {eval_txt}")

    if lstm_mdl is not None:
        export_onnx(lstm_mdl, device)

    print("\n" + "=" * 90)
    print("  ✓ Full Hardware Phase 2 Training & Evaluation Complete!")
    print(f"  Best Model: {best_model_name} (TOPSIS Rank 1, validation)")
    print("=" * 90)


# ══════════════════════════════════════════════════════════════════════════════
#  ALIASES & MAIN
# ══════════════════════════════════════════════════════════════════════════════

cell_1_run_baselines = run_cell_1_baselines
cell_2_optimize_hybrid_gate = run_cell_2_weight_optimization
cell_3_ablation_study = run_cell_3_ablation_study
cell_4_trimodal_fusion = run_cell_4_trimodal_fusion
cell_5_consolidated_topsis = run_cell_5_consolidated_topsis
run_topsis_evaluation = run_cell_5_consolidated_topsis
run_best_model_evaluation = run_cell_5_consolidated_topsis


def run_all():
    print("=" * 90)
    print("  INTELLICRASH HARDWARE — PHASE 2 MODEL TRAINING (5-STAGE PIPELINE)")
    print("=" * 90)
    run_cell_1_baselines()
    run_cell_2_weight_optimization(run_optuna=True)
    run_cell_3_ablation_study()
    run_cell_4_trimodal_fusion()
    run_cell_5_consolidated_topsis()


def main():
    run_all()


if __name__ == "__main__":
    main()
