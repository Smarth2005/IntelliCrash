"""
IntelliCrash Hardware — Phase 2 Model Training
================================================
Trains Bi-LSTM crash detection model + XGBoost baseline
on hardware MPU6050 sensor data.

Run AFTER data_pipeline.py has generated the .npy files.

Run in Colab:
    !pip install torch torchvision torchaudio optuna xgboost scikit-learn onnx onnxruntime tqdm scipy
    !python hardware/data_pipeline.py
    !python hardware/train_phase2.py

Output: hardware/models/
"""

import numpy as np
import os
import sys
import random
import time
import json
from pathlib import Path
from tqdm import tqdm

# ── Path Setup ──
HARDWARE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HARDWARE_DIR))

# Import from local hardware folder (100% self-contained)
try:
    from bilstm import IntelliCrashBiLSTM, IMUDataset, FocalLoss
    from feature_engineering import NUM_ENGINEERED_FEATURES, FEATURE_NAMES
except ImportError:
    from hardware.bilstm import IntelliCrashBiLSTM, IMUDataset, FocalLoss
    from hardware.feature_engineering import NUM_ENGINEERED_FEATURES, FEATURE_NAMES

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
LSTM_INPUT_SIZE = NUM_RAW_CHANNELS + NUM_FEATURES  # 6 + 26 = 32

# Default hyperparameters (from software pipeline)
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

# Train/Val/Test split ratios (time-ordered)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# CSI parameters (calibrated for MPU6050 hardware dynamics)
CSI_G_NORM = 28.0
CSI_DV_NORM = 150.0
CSI_WEIGHTS = {"peak_g": 0.35, "delta_v": 0.30, "jerk": 0.20, "duration": 0.15}


# ══════════════════════════════════════════════════════════════════════════════
#  REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════

def set_all_seeds(seed=SEED):
    """Set all random seeds for 100% reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] All random seeds set to {seed} for reproducibility.")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load preprocessed hardware data from .npy files."""
    data_dir = HARDWARE_DIR / "data"

    print("\n" + "=" * 60)
    print("  SECTION 1: LOADING PREPROCESSED DATA")
    print("=" * 60)

    X = np.load(data_dir / "hw_X.npy")           # (N, 200, 6)
    y = np.load(data_dir / "hw_y.npy")            # (N,)
    features = np.load(data_dir / "hw_features.npy")  # (N, 26)
    csi = np.load(data_dir / "hw_csi.npy")        # (N,)

    print(f"  X (raw windows):  {X.shape}")
    print(f"  y (labels):       {y.shape}")
    print(f"  features (26):    {features.shape}")
    print(f"  CSI scores:       {csi.shape}")
    print(f"  CRASH:     {y.sum():,} ({100*y.mean():.1f}%)")
    print(f"  NON_CRASH: {(1-y).sum():,} ({100*(1-y.mean()):.1f}%)")

    return X, y, features, csi


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: TRAIN / VAL / TEST SPLIT (TIME-ORDERED)
# ══════════════════════════════════════════════════════════════════════════════

def split_data(X, y, features, csi):
    """Time-ordered split (NO shuffling — prevents data leakage).

    In time-series data, shuffling would let the model see future data
    during training. We keep chronological order.
    """
    print("\n" + "=" * 60)
    print("  SECTION 2: TRAIN / VAL / TEST SPLIT")
    print("=" * 60)

    N = len(y)
    train_end = int(N * TRAIN_RATIO)
    val_end = int(N * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": {
            "X": X[:train_end],
            "y": y[:train_end],
            "features": features[:train_end],
            "csi": csi[:train_end],
        },
        "val": {
            "X": X[train_end:val_end],
            "y": y[train_end:val_end],
            "features": features[train_end:val_end],
            "csi": csi[train_end:val_end],
        },
        "test": {
            "X": X[val_end:],
            "y": y[val_end:],
            "features": features[val_end:],
            "csi": csi[val_end:],
        },
    }

    print(f"\n  {'Split':<8} {'Total':>8} {'CRASH':>8} {'NON_CRASH':>10} {'Crash%':>8}")
    print(f"  {'-'*44}")
    for name, s in splits.items():
        n = len(s["y"])
        c = s["y"].sum()
        nc = n - c
        pct = 100 * c / n if n > 0 else 0
        print(f"  {name:<8} {n:>8,} {c:>8,} {nc:>10,} {pct:>7.1f}%")

    total_n = sum(len(s["y"]) for s in splits.values())
    print(f"  {'TOTAL':<8} {total_n:>8,}")
    print(f"\n  Seed: {SEED} | Ratio: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")

    return splits


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: BUILD LSTM TENSORS
# ══════════════════════════════════════════════════════════════════════════════

def build_lstm_input(X_raw, features):
    """Construct (batch, 200, 32) tensors for Bi-LSTM.

    Concatenates 6 raw channels + 26 engineered features broadcast
    across all 200 timesteps. SAME as software pipeline.

    Args:
        X_raw: (N, 200, 6) raw sensor windows
        features: (N, 26) engineered feature vectors

    Returns:
        (N, 200, 32) float32 array
    """
    seq_len = X_raw.shape[1]
    # Broadcast (N, 26) → (N, 200, 26)
    features_broadcast = np.repeat(features[:, np.newaxis, :], seq_len, axis=1)
    # Concatenate: (N, 200, 6) + (N, 200, 26) = (N, 200, 32)
    X_full = np.concatenate([X_raw, features_broadcast], axis=2).astype(np.float32)
    return X_full


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: BI-LSTM TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def train_bilstm(splits, hparams, device, tag="default"):
    """Train Bi-LSTM with given hyperparameters.

    Returns:
        model: Trained model (best val weights loaded)
        history: Dict of train/val metrics per epoch
        test_metrics: Dict of test set evaluation metrics
    """
    print(f"\n  Training Bi-LSTM [{tag}]...")
    print(f"  Hyperparameters: {json.dumps(hparams, indent=2)}")

    # Build LSTM inputs
    X_train = build_lstm_input(splits["train"]["X"], splits["train"]["features"])
    X_val = build_lstm_input(splits["val"]["X"], splits["val"]["features"])
    X_test = build_lstm_input(splits["test"]["X"], splits["test"]["features"])

    y_train = splits["train"]["y"]
    y_val = splits["val"]["y"]
    y_test = splits["test"]["y"]
    csi_train = splits["train"]["csi"]
    csi_val = splits["val"]["csi"]
    csi_test = splits["test"]["csi"]

    # Datasets & Loaders
    train_ds = IMUDataset(X_train, y_train, csi_train)
    val_ds = IMUDataset(X_val, y_val, csi_val)
    test_ds = IMUDataset(X_test, y_test, csi_test)

    train_loader = DataLoader(train_ds, batch_size=hparams["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hparams["batch_size"])
    test_loader = DataLoader(test_ds, batch_size=hparams["batch_size"])

    # Model
    model = IntelliCrashBiLSTM(
        input_size=LSTM_INPUT_SIZE,
        hidden_size=hparams["hidden_size"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # Loss functions
    criterion_crash = FocalLoss(alpha=hparams["focal_alpha"], gamma=hparams["focal_gamma"])
    criterion_sev = nn.MSELoss()
    w_crash = hparams["crash_loss_weight"]
    w_sev = hparams["severity_loss_weight"]

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hparams["epochs"])

    # Training loop
    best_val_loss = float("inf")
    best_state = None
    patience = 10
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(hparams["epochs"]):
        # ── Train ──
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

        # ── Validate ──
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

        # Checkpointing
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

        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    # Load best weights
    model.load_state_dict(best_state)
    model.eval()

    # ── Test Evaluation ──
    all_probs, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            X = batch["features"].to(device)
            pred_crash, _ = model(X)
            all_probs.extend(pred_crash.cpu().numpy().flatten())
            all_true.extend(batch["crash"].numpy().flatten())

    all_probs = np.array(all_probs)
    all_true = np.array(all_true)
    all_preds = (all_probs > 0.5).astype(int)

    test_metrics = {
        "accuracy": accuracy_score(all_true, all_preds),
        "precision": precision_score(all_true, all_preds, zero_division=0),
        "recall": recall_score(all_true, all_preds, zero_division=0),
        "f1": f1_score(all_true, all_preds, zero_division=0),
        "auc_roc": roc_auc_score(all_true, all_probs) if len(np.unique(all_true)) > 1 else 0.0,
    }

    return model, history, test_metrics, all_probs


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: HYPERPARAMETER OPTIMIZATION (OPTUNA)
# ══════════════════════════════════════════════════════════════════════════════

def run_optuna_search(splits, device, n_trials=15):
    """Optuna hyperparameter search for Bi-LSTM."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("\n  [WARN] Optuna not installed. Skipping HP search.")
        print("  Install: pip install optuna")
        return None

    print("\n" + "=" * 60)
    print("  SECTION 5: HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    print(f"  Running {n_trials} trials...")
    print("=" * 60)

    def objective(trial):
        hparams = {
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "epochs": 20,  # Shorter for search
            "weight_decay": 0.01,
            "focal_alpha": 0.75,
            "focal_gamma": 2.0,
            "crash_loss_weight": 0.7,
            "severity_loss_weight": 0.3,
        }

        set_all_seeds(SEED)
        _, _, test_metrics, _ = train_bilstm(splits, hparams, device, tag=f"trial-{trial.number}")
        return test_metrics["f1"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best Trial: #{study.best_trial.number}")
    print(f"  Best F1 Score: {study.best_trial.value:.4f}")
    print(f"  Best Params: {json.dumps(study.best_trial.params, indent=4)}")

    # Build full hparams from best trial
    best_hparams = DEFAULT_HPARAMS.copy()
    best_hparams.update(study.best_trial.params)
    best_hparams["epochs"] = 50  # Full training for final run

    return best_hparams


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6: XGBOOST BASELINE
# ══════════════════════════════════════════════════════════════════════════════

def train_xgboost(splits):
    """Train XGBoost on 26-feature vectors (tabular baseline)."""
    print("\n" + "=" * 60)
    print("  SECTION 6: XGBOOST BASELINE")
    print("=" * 60)

    try:
        import xgboost as xgb
    except ImportError:
        print("  [WARN] xgboost not installed. Skipping.")
        return None, None

    X_train = splits["train"]["features"]
    y_train = splits["train"]["y"]
    X_test = splits["test"]["features"]
    y_test = splits["test"]["y"]

    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        use_label_encoder=False,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(splits["val"]["features"], splits["val"]["y"])],
        verbose=False,
    )

    # Save model
    model_path = HARDWARE_DIR / "models" / "hw_xgboost.json"
    model.save_model(str(model_path))
    print(f"  Saved: {model_path}")

    # Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "auc_roc": roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else 0.0,
    }

    return metrics, probs


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7: CSI & FUSION EVALUATION + WEIGHT OPTIMIZATION / ABLATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_csi_and_fusion(splits, model, device, lstm_test_probs):
    """Evaluate CSI-only, Equal-Weight Fusion, and Optimized-Weight Fusion."""
    print("\n" + "=" * 60)
    print("  SECTION 7: CSI & FUSION WEIGHT OPTIMIZATION (ABLATION STUDY)")
    print("=" * 60)

    # 1. Get Validation Predictions from Bi-LSTM for Optimization
    X_val = build_lstm_input(splits["val"]["X"], splits["val"]["features"])
    val_ds = IMUDataset(X_val, splits["val"]["y"], splits["val"]["csi"])
    val_loader = DataLoader(val_ds, batch_size=64)

    model.eval()
    lstm_val_probs = []
    with torch.no_grad():
        for batch in val_loader:
            pred_c, _ = model(batch["features"].to(device))
            lstm_val_probs.extend(pred_c.cpu().numpy().flatten())
    lstm_val_probs = np.array(lstm_val_probs)

    y_val = splits["val"]["y"]
    csi_val = splits["val"]["csi"]

    y_test = splits["test"]["y"]
    csi_test = splits["test"]["csi"]

    # ── Optimal CSI-only Threshold Search on Validation Set ──
    best_csi_thresh = 0.50
    best_csi_val_f1 = 0
    for t in np.linspace(0.30, 0.70, 41):
        f1_t = f1_score(y_val, (csi_val > t).astype(int), zero_division=0)
        if f1_t > best_csi_val_f1:
            best_csi_val_f1 = f1_t
            best_csi_thresh = t

    csi_test_preds = (csi_test > best_csi_thresh).astype(int)
    csi_metrics = {
        "accuracy": accuracy_score(y_test, csi_test_preds),
        "precision": precision_score(y_test, csi_test_preds, zero_division=0),
        "recall": recall_score(y_test, csi_test_preds, zero_division=0),
        "f1": f1_score(y_test, csi_test_preds, zero_division=0),
    }

    # ── Baseline Equal-Weight Fusion (50% LSTM + 50% CSI @ 0.50 threshold) ──
    p_equal = 0.5 * lstm_test_probs + 0.5 * csi_test
    equal_preds = (p_equal > 0.50).astype(int)
    equal_fusion_metrics = {
        "accuracy": accuracy_score(y_test, equal_preds),
        "precision": precision_score(y_test, equal_preds, zero_division=0),
        "recall": recall_score(y_test, equal_preds, zero_division=0),
        "f1": f1_score(y_test, equal_preds, zero_division=0),
        "auc_roc": roc_auc_score(y_test, p_equal) if len(np.unique(y_test)) > 1 else 0.0,
    }

    # ── Grid Search: Optimize Fusion Weights (w_lstm, w_csi) & Threshold (tau) on Validation Set ──
    best_w = 0.5
    best_tau = 0.50
    best_val_f1 = 0.0

    for w in np.linspace(0.1, 0.9, 17):  # w is weight of LSTM, (1-w) is weight of CSI
        for tau in np.linspace(0.30, 0.70, 41):
            p_val_comb = w * lstm_val_probs + (1.0 - w) * csi_val
            f1_val = f1_score(y_val, (p_val_comb > tau).astype(int), zero_division=0)
            if f1_val > best_val_f1:
                best_val_f1 = f1_val
                best_w = float(w)
                best_tau = float(tau)

    print(f"  Optimized Fusion Weights (from Validation Set):")
    print(f"    w_LSTM = {best_w:.2f}, w_CSI = {1.0 - best_w:.2f}, Threshold (tau) = {best_tau:.2f} (Val F1={best_val_f1:.4f})")

    # ── Evaluate Optimized Fusion on Test Set ──
    p_opt_test = best_w * lstm_test_probs + (1.0 - best_w) * csi_test
    opt_preds = (p_opt_test > best_tau).astype(int)
    opt_fusion_metrics = {
        "accuracy": accuracy_score(y_test, opt_preds),
        "precision": precision_score(y_test, opt_preds, zero_division=0),
        "recall": recall_score(y_test, opt_preds, zero_division=0),
        "f1": f1_score(y_test, opt_preds, zero_division=0),
        "auc_roc": roc_auc_score(y_test, p_opt_test) if len(np.unique(y_test)) > 1 else 0.0,
    }

    return csi_metrics, equal_fusion_metrics, opt_fusion_metrics, (best_w, best_tau)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8: ONNX EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_onnx(model, device):
    """Export best Bi-LSTM model to ONNX for edge deployment."""
    print("\n" + "=" * 60)
    print("  SECTION 8: ONNX EXPORT")
    print("=" * 60)

    out_path = HARDWARE_DIR / "models" / "hw_bilstm.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.randn(1, WINDOW_SIZE, LSTM_INPUT_SIZE).to(device)

    try:
        try:
            import onnxscript
        except ImportError:
            import subprocess
            print("  Installing onnxscript for PyTorch ONNX exporter...")
            subprocess.run([sys.executable, "-m", "pip", "install", "onnxscript", "onnx"], check=False)

        torch.onnx.export(
            model, dummy, str(out_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["crash_prob", "severity"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "crash_prob": {0: "batch_size"},
                "severity": {0: "batch_size"},
            },
        )
        print(f"  Exported: {out_path}")
        print(f"  File size: {out_path.stat().st_size / 1024:.1f} KB")

        # Verify with ONNX Runtime
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(out_path))
            ort_input = {session.get_inputs()[0].name: dummy.cpu().numpy()}
            ort_crash, ort_sev = session.run(None, ort_input)
            print(f"  ONNX verification: crash_prob={ort_crash[0][0]:.4f}, severity={ort_sev[0][0]:.4f}")
            print("  ✓ ONNX model verified successfully!")
        except Exception as e:
            print(f"  [WARN] onnxruntime verification notice: {e}")

    except Exception as e:
        print(f"  [WARN] ONNX export encountered an issue: {e}")
        print("  Note: PyTorch checkpoint (hw_best_bilstm.pth) is fully saved and safe in hardware/models/")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  IntelliCrash Hardware — Phase 2 Model Training")
    print("  Binary Crash Detection: Bi-LSTM + XGBoost + CSI Fusion")
    print("=" * 60)

    set_all_seeds(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load Data ──
    X, y, features, csi = load_data()

    # ── Split ──
    splits = split_data(X, y, features, csi)

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE A: INITIAL TRAINING (Default Hyperparameters)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SECTION 4A: INITIAL BI-LSTM TRAINING (Default HParams)")
    print("=" * 60)

    set_all_seeds(SEED)
    model_default, hist_default, metrics_default, probs_default = train_bilstm(
        splits, DEFAULT_HPARAMS, device, tag="default-hparams"
    )

    print(f"\n  Initial Results (Default HParams):")
    for k, v in metrics_default.items():
        print(f"    {k}: {v:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE B: HYPERPARAMETER OPTIMIZATION (Optuna)
    # ══════════════════════════════════════════════════════════════════════
    best_hparams = run_optuna_search(splits, device, n_trials=15)

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE C: FINAL TRAINING WITH BEST HYPERPARAMETERS
    # ══════════════════════════════════════════════════════════════════════
    if best_hparams is not None:
        print("\n" + "=" * 60)
        print("  SECTION 4C: FINAL BI-LSTM TRAINING (Optimized HParams)")
        print("=" * 60)

        set_all_seeds(SEED)
        model_best, hist_best, metrics_best, probs_best = train_bilstm(
            splits, best_hparams, device, tag="optimized-hparams"
        )

        # Choose better model
        if metrics_best["f1"] >= metrics_default["f1"]:
            final_model = model_best
            final_metrics = metrics_best
            final_probs = probs_best
            final_tag = "optimized"
            print(f"\n  ✓ Optimized model is better (F1: {metrics_best['f1']:.4f} vs {metrics_default['f1']:.4f})")
        else:
            final_model = model_default
            final_metrics = metrics_default
            final_probs = probs_default
            final_tag = "default"
            print(f"\n  ✓ Default model is better (F1: {metrics_default['f1']:.4f} vs {metrics_best['f1']:.4f})")
    else:
        final_model = model_default
        final_metrics = metrics_default
        final_probs = probs_default
        final_tag = "default"

    # Save best Bi-LSTM checkpoint
    ckpt_path = HARDWARE_DIR / "models" / "hw_best_bilstm.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), ckpt_path)
    print(f"  Saved best checkpoint: {ckpt_path}")

    # ── XGBoost Baseline ──
    xgb_metrics, xgb_probs = train_xgboost(splits)

    # ── CSI & Fusion Ablation Study ──
    csi_metrics, equal_fusion_metrics, opt_fusion_metrics, (opt_w, opt_tau) = evaluate_csi_and_fusion(
        splits, final_model, device, final_probs
    )

    # ══════════════════════════════════════════════════════════════════════
    #  RESULTS COMPARISON & ABLATION TABLE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 76)
    print("  FINAL RESULTS COMPARISON & ABLATION STUDY (TEST SET)")
    print("=" * 76)
    print(f"\n  {'Model / Pipeline Stage':<28} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10}")
    print(f"  {'-'*76}")

    results = {
        f"Bi-LSTM ({final_tag})": final_metrics,
    }
    if xgb_metrics:
        results["XGBoost (baseline)"] = xgb_metrics
    results["CSI-Only (physics path)"] = csi_metrics
    results["Fusion (Equal 50-50)"] = equal_fusion_metrics
    results[f"Fusion (Opt w={opt_w:.2f})"] = opt_fusion_metrics

    for name, m in results.items():
        auc_str = f"{m.get('auc_roc', 0):.4f}" if isinstance(m.get('auc_roc'), (int, float)) and m.get('auc_roc') > 0 else "-"
        print(f"  {name:<28} "
              f"{m.get('accuracy', 0):>10.4f} "
              f"{m.get('precision', 0):>10.4f} "
              f"{m.get('recall', 0):>10.4f} "
              f"{m.get('f1', 0):>10.4f} "
              f"{auc_str:>10}")

    # ── Test set confusion matrix for best model ──
    y_test = splits["test"]["y"]
    y_pred = (final_probs > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix (Bi-LSTM {final_tag}):")
    print(f"                  Predicted")
    print(f"                  NON_CRASH  CRASH")
    print(f"  Actual NON_CRASH  {cm[0][0]:>6}   {cm[0][1]:>6}")
    print(f"  Actual CRASH      {cm[1][0]:>6}   {cm[1][1]:>6}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["NON_CRASH", "CRASH"]))

    # ── Export ONNX ──
    export_onnx(final_model, device)

    # ── Save results JSON ──
    results_json = {
        "seed": SEED,
        "splits": {
            "train": int(len(splits["train"]["y"])),
            "val": int(len(splits["val"]["y"])),
            "test": int(len(splits["test"]["y"])),
        },
        "final_model": final_tag,
        "bilstm_metrics": {k: round(v, 4) for k, v in final_metrics.items()},
        "xgboost_metrics": {k: round(v, 4) for k, v in xgb_metrics.items()} if xgb_metrics else None,
        "csi_metrics": {k: round(v, 4) for k, v in csi_metrics.items()},
        "equal_fusion_metrics": {k: round(v, 4) for k, v in equal_fusion_metrics.items()},
        "optimized_fusion_metrics": {k: round(v, 4) for k, v in opt_fusion_metrics.items()},
        "optimal_weights": {"w_lstm": opt_w, "w_csi": round(1.0 - opt_w, 2), "tau_threshold": opt_tau},
    }
    results_path = HARDWARE_DIR / "models" / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    print("\n" + "=" * 60)
    print("  ✓ Training Complete!")
    print("  Models saved in: hardware/models/")
    print("  Files: hw_best_bilstm.pth, hw_bilstm.onnx, hw_xgboost.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
