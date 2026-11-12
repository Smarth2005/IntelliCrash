"""
IntelliCrash — Hybrid Fusion Gate Ablation Study.

Compares three architecture configurations:
1. Physics CSI Gate Only
2. Bi-LSTM Only
3. Fused Hybrid (Bi-LSTM + CSI) at the specified weights

Output:
- Console table
- outputs/reports/ablation_study.csv
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config
from src.models.bilstm import IntelliCrashBiLSTM
from src.features.feature_engineering import compute_csi

def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate Accuracy, Recall, False Positive Rate (FPR), F1-Score, and AUC-ROC"""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1 = f1_score(y_true, y_pred) if (tp + fp + fn) > 0 else 0
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = 0.5
    
    return accuracy * 100, recall * 100, fpr * 100, f1 * 100, auc

def run_ablation_study(w_ml, w_csi):
    print("Loading test data...")
    data_dir = Path("data/processed/windows")
    
    try:
        X_test = np.load(data_dir / "imu_augmented_X_test.npy")
        y_test = np.load(data_dir / "imu_augmented_y_test.npy")
    except FileNotFoundError:
        print("Test data not found. Please ensure preprocess_imu.py has been run and test sets are saved.")
        return

    # Extract the 26 engineered features (last 26 columns in the 32-channel input)
    features_test = X_test[:, 0, 6:] 

    print("Loading trained Bi-LSTM model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntelliCrashBiLSTM(input_size=32, hidden_size=128, num_layers=2).to(device)
    
    try:
        model.load_state_dict(torch.load("models/checkpoints/best_bilstm.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        print("Model checkpoint not found. Please train the LSTM first.")
        return

    print("Running Inference on Test Set...\n")
    
    # 1. Get LSTM Predictions (batched)
    batch_size = 256
    lstm_probs = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            probs, _ = model(batch)
            lstm_probs.append(probs.cpu().numpy().flatten())
    lstm_probs = np.concatenate(lstm_probs)
    
    # 2. Get Physics CSI Predictions
    csi_scores = compute_csi(features_test, mode="real_car")
    
    # 3. Calculate Fusion Scores
    fusion_scores = (w_ml * lstm_probs) + (w_csi * csi_scores)

    # Thresholding
    pred_lstm = (lstm_probs > 0.5).astype(int)
    pred_physics = (csi_scores > 0.5).astype(int)
    pred_fusion = (fusion_scores > 0.5).astype(int)

    # Calculate Metrics
    acc_lstm, recall_lstm, fpr_lstm, f1_lstm, auc_lstm = calculate_metrics(y_test, pred_lstm, lstm_probs)
    acc_phys, recall_phys, fpr_phys, f1_phys, auc_phys = calculate_metrics(y_test, pred_physics, csi_scores)
    acc_fuse, recall_fuse, fpr_fuse, f1_fuse, auc_fuse = calculate_metrics(y_test, pred_fusion, fusion_scores)

    # Print Table
    print("===================================================================================================================")
    print("  Ablation Study of the Hybrid Fusion Gate")
    print("===================================================================================================================")
    print(f"{'Architecture Configuration':<30} | {'Accuracy':<10} | {'Recall':<10} | {'FPR':<10} | {'F1-Score':<10} | {'AUC-ROC':<10}")
    print("-" * 115)
    print(f"{'Physics CSI Gate Only':<30} | {acc_phys:>6.2f}%     | {recall_phys:>6.2f}%     | {fpr_phys:>6.2f}%     | {f1_phys:>6.2f}%     | {auc_phys:>6.4f}")
    print(f"{'Bi-LSTM Only':<30} | {acc_lstm:>6.2f}%     | {recall_lstm:>6.2f}%     | {fpr_lstm:>6.2f}%     | {f1_lstm:>6.2f}%     | {auc_lstm:>6.4f}")
    print(f"{'Fused (Hybrid Bi-LSTM)':<30} | {acc_fuse:>6.2f}%     | {recall_fuse:>6.2f}%     | {fpr_fuse:>6.2f}%     | {f1_fuse:>6.2f}%     | {auc_fuse:>6.4f}")
    print("===================================================================================================================")

    # Save CSV
    cfg = get_config()
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([
        {"Architecture Configuration": "Physics CSI Gate Only", "Accuracy (%)": round(acc_phys, 2), "Recall (%)": round(recall_phys, 2), "FPR (%)": round(fpr_phys, 2), "F1-Score (%)": round(f1_phys, 2), "AUC-ROC": round(auc_phys, 4)},
        {"Architecture Configuration": "Bi-LSTM Only", "Accuracy (%)": round(acc_lstm, 2), "Recall (%)": round(recall_lstm, 2), "FPR (%)": round(fpr_lstm, 2), "F1-Score (%)": round(f1_lstm, 2), "AUC-ROC": round(auc_lstm, 4)},
        {"Architecture Configuration": f"Fused Hybrid ({w_ml:.1f} ML + {w_csi:.1f} CSI)", "Accuracy (%)": round(acc_fuse, 2), "Recall (%)": round(recall_fuse, 2), "FPR (%)": round(fpr_fuse, 2), "F1-Score (%)": round(f1_fuse, 2), "AUC-ROC": round(auc_fuse, 4)},
    ])
    csv_path = reports_dir / "ablation_study.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n=> Table saved to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Architecture Ablation Study")
    parser.add_argument("--w_ml", type=float, default=None, help="Weight for ML confidence (auto-detected from TOPSIS if not specified)")
    parser.add_argument("--w_csi", type=float, default=None, help="Weight for CSI score (auto-detected from TOPSIS if not specified)")
    args = parser.parse_args()
    
    # Auto-detect TOPSIS-optimized weights if not manually specified
    if args.w_ml is None or args.w_csi is None:
        import json
        cfg = get_config()
        weights_path = Path(cfg["paths"]["reports_dir"]) / "optimal_fusion_weights.json"
        try:
            with open(weights_path) as f:
                opt = json.load(f)
            w_ml, w_csi = opt["w_ml"], opt["w_csi"]
            print(f"\n[Ablation] Auto-loaded TOPSIS-optimized weights: {w_ml} ML + {w_csi} CSI\n")
        except FileNotFoundError:
            w_ml, w_csi = 0.7, 0.3
            print(f"\n[Ablation] WARNING: optimal_fusion_weights.json not found. Using default {w_ml}:{w_csi}\n")
    else:
        w_ml, w_csi = args.w_ml, args.w_csi
        print(f"\n[Ablation] Using manually specified weights: {w_ml} ML + {w_csi} CSI\n")
    
    run_ablation_study(w_ml, w_csi)
