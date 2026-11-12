"""
IntelliCrash — Real Crash Data (VZCrash) Evaluation Script.

This script evaluates the trained Bi-LSTM model and the Hybrid Fusion Gate
on the real-world VZCrash validation set (streamed and prepared by download_vzcrash.py).

Outputs:
- Metrics (Accuracy, Recall, FPR, F1-Score, AUC-ROC) for:
  1. Bi-LSTM Only
  2. Physics CSI Only
  3. Fused Hybrid (Bi-LSTM + CSI)
- Detailed validation summary report saved to outputs/reports/vzcrash_validation.csv
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

# Setup project paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config
from src.models.bilstm import IntelliCrashBiLSTM
from src.features.feature_engineering import compute_csi

def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate standard safety-critical metrics."""
    cm = confusion_matrix(y_true, y_pred)
    # Ensure shape is correct (binary classification)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback for single-class predictions
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
    accuracy = accuracy_score(y_true, y_pred)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1 = f1_score(y_true, y_pred) if (tp + fp + fn) > 0 else 0
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = 0.5
    
    return accuracy * 100, recall * 100, fpr * 100, f1 * 100, auc

def main():
    print("--- IntelliCrash VZCrash Validation Evaluator ---")
    cfg = get_config()
    
    # Paths
    processed_dir = Path(cfg["paths"]["processed_dir"])
    val_dir = processed_dir / "vz_validation"
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load VZCrash Validation Data
    print("Loading VZCrash validation arrays...")
    try:
        X_val = np.load(val_dir / "vz_X.npy")
        y_val = np.load(val_dir / "vz_y.npy")
        features_val = np.load(val_dir / "vz_features.npy")
        csi_scores = np.load(val_dir / "vz_csi.npy")
    except FileNotFoundError:
        print("\n❌ Error: Validation arrays not found. Please run the download script first:")
        print("   python src/data/download_vzcrash.py")
        sys.exit(1)
        
    print(f"📊 Loaded {len(y_val):,} validation windows.")
    print(f"   - Crash windows    : {np.sum(y_val == 1):,}")
    print(f"   - Non-crash windows: {np.sum(y_val == 0):,}")
    
    # 2. Prepare LSTM Inputs
    # Broaden 26 engineered features over 200 time steps and merge with 6 raw channels
    print("\nPreparing model inputs (depth-concatenating raw signals & expanded features)...")
    seq_len = X_val.shape[1]
    features_expanded = np.repeat(features_val[:, np.newaxis, :], seq_len, axis=1)
    X_full = np.concatenate([X_val, features_expanded], axis=2)  # Shape: (N, 200, 32)
    
    # 3. Load Model
    print("\nLoading trained Bi-LSTM model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntelliCrashBiLSTM(input_size=32, hidden_size=128, num_layers=2).to(device)
    
    try:
        checkpoint_path = Path(cfg["paths"]["model_checkpoints"]) / "best_bilstm.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print("✅ Deployed model checkpoint loaded successfully!")
    except FileNotFoundError:
        print("\n❌ Error: Model checkpoint not found. Please train the model first:")
        print("   python src/models/train_bilstm.py")
        sys.exit(1)
        
    # 4. Run Model Inference
    print("\nRunning neural network inference on validation windows...")
    batch_size = 256
    lstm_probs = []
    
    with torch.no_grad():
        for i in range(0, len(X_full), batch_size):
            batch = torch.FloatTensor(X_full[i:i+batch_size]).to(device)
            probs, _ = model(batch)
            lstm_probs.extend(probs.cpu().numpy().flatten())
            
    lstm_probs = np.array(lstm_probs, dtype=np.float32)
    
    # 5. Load Optimized Fusion Weights
    weights_path = reports_dir / "optimal_fusion_weights.json"
    try:
        with open(weights_path) as f:
            opt = json.load(f)
        w_ml, w_csi = opt["w_ml"], opt["w_csi"]
        print(f"ℹ️ Found optimal fusion weights: {w_ml} ML + {w_csi} CSI")
    except FileNotFoundError:
        w_ml, w_csi = 0.5, 0.5
        print(f"⚠️ Warning: Optimal weights file not found. Using default {w_ml}:{w_csi}")
        
    # 6. Compute Fusion Scores
    fusion_scores = (w_ml * lstm_probs) + (w_csi * csi_scores)
    
    # 6.5 Threshold Sweep Optimization
    print("\n🔍 Optimizing Decision Thresholds (sweeping 0.01 to 0.99 to maximize F1-Score)...")
    def find_best_threshold(y_true, scores):
        best_th = 0.5
        best_f1 = -1
        for th in np.linspace(0.01, 0.99, 99):
            preds = (scores > th).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        return best_th, best_f1

    th_lstm, _ = find_best_threshold(y_val, lstm_probs)
    th_phys, _ = find_best_threshold(y_val, csi_scores)
    th_fuse, _ = find_best_threshold(y_val, fusion_scores)
    
    print(f"   - Optimal LSTM Threshold   : {th_lstm:.2f}")
    print(f"   - Optimal Physics Threshold: {th_phys:.2f}")
    print(f"   - Optimal Fusion Threshold : {th_fuse:.2f}")

    # Thresholding (Default 0.5)
    pred_lstm = (lstm_probs > 0.5).astype(int)
    pred_physics = (csi_scores > 0.5).astype(int)
    pred_fusion = (fusion_scores > 0.5).astype(int)
    
    # Thresholding (Optimized)
    pred_lstm_opt = (lstm_probs > th_lstm).astype(int)
    pred_physics_opt = (csi_scores > th_phys).astype(int)
    pred_fusion_opt = (fusion_scores > th_fuse).astype(int)
    
    # 7. Evaluate Configurations (Default 0.5)
    acc_lstm, rec_lstm, fpr_lstm, f1_lstm, auc_lstm = calculate_metrics(y_val, pred_lstm, lstm_probs)
    acc_phys, rec_phys, fpr_phys, f1_phys, auc_phys = calculate_metrics(y_val, pred_physics, csi_scores)
    acc_fuse, rec_fuse, fpr_fuse, f1_fuse, auc_fuse = calculate_metrics(y_val, pred_fusion, fusion_scores)
    
    # 7.5 Evaluate Configurations (Optimized)
    acc_lstm_o, rec_lstm_o, fpr_lstm_o, f1_lstm_o, _ = calculate_metrics(y_val, pred_lstm_opt, lstm_probs)
    acc_phys_o, rec_phys_o, fpr_phys_o, f1_phys_o, _ = calculate_metrics(y_val, pred_physics_opt, csi_scores)
    acc_fuse_o, rec_fuse_o, fpr_fuse_o, f1_fuse_o, _ = calculate_metrics(y_val, pred_fusion_opt, fusion_scores)

    # Display results table
    print("\n=========================================================================================")
    print("  Real Crash Data Validation Results (VZCrash) — Default Threshold (0.50)")
    print("=========================================================================================")
    print(f"{'Configuration':<35} | {'Accuracy':<10} | {'Recall':<8} | {'FPR':<8} | {'F1-Score':<10} | {'AUC-ROC':<8}")
    print("-----------------------------------------------------------------------------------------")
    print(f"{'1. Bi-LSTM Neural Path (1.0:0.0)':<35} | {acc_lstm:<9.2f}% | {rec_lstm:<7.2f}% | {fpr_lstm:<7.2f}% | {f1_lstm:<9.2f}% | {auc_lstm:<8.4f}")
    print(f"{'2. Physics CSI Path (0.0:1.0)':<35} | {acc_phys:<9.2f}% | {rec_phys:<7.2f}% | {fpr_phys:<7.2f}% | {f1_phys:<9.2f}% | {auc_phys:<8.4f}")
    print(f"{'3. Fused Hybrid Gate ('+str(w_ml)+':'+str(w_csi)+') ★':<35} | {acc_fuse:<9.2f}% | {rec_fuse:<7.2f}% | {fpr_fuse:<7.2f}% | {f1_fuse:<9.2f}% | {auc_fuse:<8.4f}")
    print("=========================================================================================")

    print("\n=========================================================================================")
    print("  Real Crash Data Validation Results (VZCrash) — Optimized Thresholds")
    print("=========================================================================================")
    print(f"{'Configuration (Threshold)':<35} | {'Accuracy':<10} | {'Recall':<8} | {'FPR':<8} | {'F1-Score':<10} | {'AUC-ROC':<8}")
    print("-----------------------------------------------------------------------------------------")
    print(f"{'1. Bi-LSTM Neural Path ('+f'{th_lstm:.2f}'+')':<35} | {acc_lstm_o:<9.2f}% | {rec_lstm_o:<7.2f}% | {fpr_lstm_o:<7.2f}% | {f1_lstm_o:<9.2f}% | {auc_lstm:<8.4f}")
    print(f"{'2. Physics CSI Path ('+f'{th_phys:.2f}'+')':<35} | {acc_phys_o:<9.2f}% | {rec_phys_o:<7.2f}% | {fpr_phys_o:<7.2f}% | {f1_phys_o:<9.2f}% | {auc_phys:<8.4f}")
    print(f"{'3. Fused Hybrid Gate ('+f'{th_fuse:.2f}'+') ★':<35} | {acc_fuse_o:<9.2f}% | {rec_fuse_o:<7.2f}% | {fpr_fuse_o:<7.2f}% | {f1_fuse_o:<9.2f}% | {auc_fuse:<8.4f}")
    print("=========================================================================================")
    
    # 8. Save Report
    report_df = pd.DataFrame([
        {
            "configuration": "Bi-LSTM Neural Path",
            "accuracy": acc_lstm,
            "recall": rec_lstm,
            "fpr": fpr_lstm,
            "f1_score": f1_lstm,
            "auc_roc": auc_lstm
        },
        {
            "configuration": "Physics CSI Path",
            "accuracy": acc_phys,
            "recall": rec_phys,
            "fpr": fpr_phys,
            "f1_score": f1_phys,
            "auc_roc": auc_phys
        },
        {
            "configuration": "Fused Hybrid Gate",
            "accuracy": acc_fuse,
            "recall": rec_fuse,
            "fpr": fpr_fuse,
            "f1_score": f1_fuse,
            "auc_roc": auc_fuse
        }
    ])
    report_path = reports_dir / "vzcrash_validation_report.csv"
    report_df.to_csv(report_path, index=False)
    print(f"\n📝 Detailed validation report saved to {report_path.resolve()}")
    
    # 9. Print Confusion Matrix for Fusion Gate
    cm = confusion_matrix(y_val, pred_fusion)
    print("\n🧩 Fused Hybrid Gate Confusion Matrix:")
    print(f"             Predicted Non-Crash  Predicted Crash")
    print(f"Actual Non-Crash:    {cm[0,0]:<15} {cm[0,1]}")
    print(f"Actual Crash:        {cm[1,0]:<15} {cm[1,1]}")
    
if __name__ == "__main__":
    main()
