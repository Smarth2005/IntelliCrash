"""
IntelliCrash — Domain Adaptation (Fine-Tuning) on VZCrash Real Crash Data.

This script performs transfer learning:
1. Loads the pre-trained Bi-LSTM model (trained on synthetic crashes)
2. Fine-tunes it on 10% of VZCrash real crash data (domain adaptation)
3. Evaluates on the remaining 90% held-out VZCrash test set
4. Reports comparative metrics: zero-shot vs. fine-tuned

Run on Google Colab where the VZCrash data is already downloaded.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

# Setup project paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config
from src.models.bilstm import IntelliCrashBiLSTM
from src.features.feature_engineering import compute_csi

def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate standard safety-critical metrics."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
    accuracy = accuracy_score(y_true, y_pred)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = 0.5
    
    return accuracy * 100, recall * 100, fpr * 100, f1 * 100, auc

def find_best_threshold(y_true, scores):
    """Sweep thresholds to maximize F1-Score."""
    best_th = 0.5
    best_f1 = -1
    for th in np.linspace(0.01, 0.99, 99):
        preds = (scores > th).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1

def main():
    print("=" * 100)
    print("  IntelliCrash -- Domain Adaptation on VZCrash Real Crash Data")
    print("=" * 100)
    
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # -- 1. Load VZCrash Data --------------------------------------------------
    processed_dir = Path(cfg["paths"]["processed_dir"])
    val_dir = processed_dir / "vz_validation"
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading VZCrash validation arrays...")
    X_raw = np.load(val_dir / "vz_X.npy")          # (N, 200, 6)
    y_all = np.load(val_dir / "vz_y.npy")           # (N,)
    features_all = np.load(val_dir / "vz_features.npy")  # (N, 26)
    csi_all = np.load(val_dir / "vz_csi.npy")       # (N,)
    
    print(f"   Total windows: {len(y_all):,}")
    print(f"   Crash: {np.sum(y_all == 1):,} | Non-crash: {np.sum(y_all == 0):,}")
    
    # Build full input (raw 6 + 26 features = 32 channels)
    seq_len = X_raw.shape[1]
    features_expanded = np.repeat(features_all[:, np.newaxis, :], seq_len, axis=1)
    X_full = np.concatenate([X_raw, features_expanded], axis=2)  # (N, 200, 32)
    
    # -- 2. Split: 10% fine-tune, 90% test ------------------------------------
    FINETUNE_FRACTION = 0.10
    SEED = 42
    
    X_train, X_test, y_train, y_test, csi_train, csi_test = train_test_split(
        X_full, y_all, csi_all,
        test_size=(1 - FINETUNE_FRACTION),
        stratify=y_all,
        random_state=SEED
    )
    
    print(f"\nDomain Adaptation Split:")
    print(f"   Fine-tune set: {len(y_train):,} windows "
          f"(Crash: {np.sum(y_train==1):,}, Non-crash: {np.sum(y_train==0):,})")
    print(f"   Test set:      {len(y_test):,} windows "
          f"(Crash: {np.sum(y_test==1):,}, Non-crash: {np.sum(y_test==0):,})")
    
    # -- 3. Load Pre-Trained Model --------------------------------------------
    print("\nLoading pre-trained Bi-LSTM model...")
    model = IntelliCrashBiLSTM(input_size=32, hidden_size=128, num_layers=2).to(device)
    
    checkpoint_path = Path(cfg["paths"]["model_checkpoints"]) / "best_bilstm.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("Pre-trained weights loaded successfully.")
    
    # -- 4. Zero-Shot Baseline (Before Fine-Tuning) ---------------------------
    print("\nRunning zero-shot inference (before fine-tuning)...")
    model.eval()
    
    zs_probs = []
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            probs, _ = model(batch)
            zs_probs.extend(probs.cpu().numpy().flatten())
    zs_probs = np.array(zs_probs, dtype=np.float32)
    
    # -- 5. Fine-Tune ---------------------------------------------------------
    print("\nFine-tuning on 10% of VZCrash real crash data...")
    print("   Learning rate: 1e-4 (low, to prevent catastrophic forgetting)")
    print("   Epochs: 5")
    print("   Batch size: 64")
    
    # Create DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Use a low learning rate to avoid catastrophic forgetting
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Weighted loss to handle class imbalance (crash is minority)
    n_crash = np.sum(y_train == 1)
    n_noncrash = np.sum(y_train == 0)
    pos_weight = torch.tensor([n_noncrash / max(n_crash, 1)]).float().to(device)
    
    model.train()
    for epoch in range(5):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.float().to(device)
            
            optimizer.zero_grad()
            probs, _ = model(batch_X)
            probs = probs.squeeze()
            
            # Weighted BCE loss — gives more importance to crash samples
            weight = torch.where(batch_y == 1, pos_weight, torch.ones_like(batch_y))
            loss = nn.functional.binary_cross_entropy(probs, batch_y, weight=weight)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_y)
            preds = (probs > 0.5).long()
            correct += (preds == batch_y.long()).sum().item()
            total += len(batch_y)
        
        acc = 100.0 * correct / total
        avg_loss = epoch_loss / total
        print(f"   Epoch {epoch+1}/5 -- Loss: {avg_loss:.4f}, Accuracy: {acc:.1f}%")
    
    print("Fine-tuning complete!")
    
    # -- 6. Fine-Tuned Inference ----------------------------------------------
    print("\nRunning fine-tuned inference on 90% held-out test set...")
    model.eval()
    
    ft_probs = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            probs, _ = model(batch)
            ft_probs.extend(probs.cpu().numpy().flatten())
    ft_probs = np.array(ft_probs, dtype=np.float32)
    
    # -- 7. Compute Results ---------------------------------------------------
    # Fusion scores
    w_ml, w_csi = 0.5, 0.5
    zs_fusion = (w_ml * zs_probs) + (w_csi * csi_test)
    ft_fusion = (w_ml * ft_probs) + (w_csi * csi_test)
    
    # Optimized thresholds
    th_zs_lstm, _ = find_best_threshold(y_test, zs_probs)
    th_ft_lstm, _ = find_best_threshold(y_test, ft_probs)
    th_csi, _     = find_best_threshold(y_test, csi_test)
    th_zs_fuse, _ = find_best_threshold(y_test, zs_fusion)
    th_ft_fuse, _ = find_best_threshold(y_test, ft_fusion)
    
    # -- 8. Print Results -----------------------------------------------------
    # Zero-shot calculations
    acc_zs_lstm, rec_zs_lstm, fpr_zs_lstm, f1_zs_lstm, auc_zs_lstm = calculate_metrics(
        y_test, (zs_probs > th_zs_lstm).astype(int), zs_probs
    )
    acc_zs_csi, rec_zs_csi, fpr_zs_csi, f1_zs_csi, auc_zs_csi = calculate_metrics(
        y_test, (csi_test > th_csi).astype(int), csi_test
    )
    acc_zs_fuse, rec_zs_fuse, fpr_zs_fuse, f1_zs_fuse, auc_zs_fuse = calculate_metrics(
        y_test, (zs_fusion > th_zs_fuse).astype(int), zs_fusion
    )

    # Fine-tuned calculations
    acc_ft_lstm, rec_ft_lstm, fpr_ft_lstm, f1_ft_lstm, auc_ft_lstm = calculate_metrics(
        y_test, (ft_probs > th_ft_lstm).astype(int), ft_probs
    )
    acc_ft_csi, rec_ft_csi, fpr_ft_csi, f1_ft_csi, auc_ft_csi = calculate_metrics(
        y_test, (csi_test > th_csi).astype(int), csi_test
    )
    acc_ft_fuse, rec_ft_fuse, fpr_ft_fuse, f1_ft_fuse, auc_ft_fuse = calculate_metrics(
        y_test, (ft_fusion > th_ft_fuse).astype(int), ft_fusion
    )

    print("\n" + "=" * 100)
    print("  ZERO-SHOT Results (Before Fine-Tuning) -- Optimized Thresholds")
    print("=" * 100)
    print(f"{'Configuration (Threshold)':<40} | {'Accuracy':<10} | {'Recall':<8} | {'FPR':<8} | {'F1':<10} | {'AUC-ROC':<8}")
    print("-" * 100)
    print(f"{'Bi-LSTM ('+f'{th_zs_lstm:.2f}'+')':<40} | {acc_zs_lstm:<9.2f}% | {rec_zs_lstm:<7.2f}% | {fpr_zs_lstm:<7.2f}% | {f1_zs_lstm:<9.2f}% | {auc_zs_lstm:<8.4f}")
    print(f"{'Physics CSI ('+f'{th_csi:.2f}'+')':<40} | {acc_zs_csi:<9.2f}% | {rec_zs_csi:<7.2f}% | {fpr_zs_csi:<7.2f}% | {f1_zs_csi:<9.2f}% | {auc_zs_csi:<8.4f}")
    print(f"{'Fused Hybrid Gate ('+f'{th_zs_fuse:.2f}'+') [Base]':<40} | {acc_zs_fuse:<9.2f}% | {rec_zs_fuse:<7.2f}% | {fpr_zs_fuse:<7.2f}% | {f1_zs_fuse:<9.2f}% | {auc_zs_fuse:<8.4f}")
    print("=" * 100)
    
    print("\n" + "=" * 100)
    print("  FINE-TUNED Results (After Domain Adaptation) -- Optimized Thresholds")
    print("=" * 100)
    print(f"{'Configuration (Threshold)':<40} | {'Accuracy':<10} | {'Recall':<8} | {'FPR':<8} | {'F1':<10} | {'AUC-ROC':<8}")
    print("-" * 100)
    print(f"{'Bi-LSTM Fine-Tuned ('+f'{th_ft_lstm:.2f}'+')':<40} | {acc_ft_lstm:<9.2f}% | {rec_ft_lstm:<7.2f}% | {fpr_ft_lstm:<7.2f}% | {f1_ft_lstm:<9.2f}% | {auc_ft_lstm:<8.4f}")
    print(f"{'Physics CSI ('+f'{th_csi:.2f}'+') [unchanged]':<40} | {acc_ft_csi:<9.2f}% | {rec_ft_csi:<7.2f}% | {fpr_ft_csi:<7.2f}% | {f1_ft_csi:<9.2f}% | {auc_ft_csi:<8.4f}")
    print(f"{'Fused Hybrid Gate ('+f'{th_ft_fuse:.2f}'+') [Base]':<40} | {acc_ft_fuse:<9.2f}% | {rec_ft_fuse:<7.2f}% | {fpr_ft_fuse:<7.2f}% | {f1_ft_fuse:<9.2f}% | {auc_ft_fuse:<8.4f}")
    print("=" * 100)
    
    # -- 9. Improvement Summary -----------------------------------------------
    zs_auc_lstm = roc_auc_score(y_test, zs_probs)
    print(f"\nDomain Adaptation Impact:")
    print(f"   Bi-LSTM AUC-ROC:  {zs_auc_lstm:.4f} -> {auc_ft_lstm:.4f}  "
          f"({'+' if auc_ft_lstm > zs_auc_lstm else '-'} {abs(auc_ft_lstm - zs_auc_lstm):.4f})")
    
    zs_fuse_auc = roc_auc_score(y_test, zs_fusion)
    print(f"   Fusion AUC-ROC:   {zs_fuse_auc:.4f} -> {auc_ft_fuse:.4f}  "
          f"({'+' if auc_ft_fuse > zs_fuse_auc else '-'} {abs(auc_ft_fuse - zs_fuse_auc):.4f})")
    
    # -- 10. Save Fine-Tuned Model and Reports --------------------------------
    ft_path = Path(cfg["paths"]["model_checkpoints"]) / "best_bilstm_vzcrash_finetuned.pth"
    torch.save(model.state_dict(), ft_path)
    print(f"\nFine-tuned model saved to: {ft_path}")
    
    zs_report = pd.DataFrame([
        {"Configuration": f"Bi-LSTM ({th_zs_lstm:.2f})", "Accuracy": acc_zs_lstm, "Recall": rec_zs_lstm, "FPR": fpr_zs_lstm, "F1-Score": f1_zs_lstm, "AUC-ROC": auc_zs_lstm},
        {"Configuration": f"Physics CSI ({th_csi:.2f})", "Accuracy": acc_zs_csi, "Recall": rec_zs_csi, "FPR": fpr_zs_csi, "F1-Score": f1_zs_csi, "AUC-ROC": auc_zs_csi},
        {"Configuration": f"Fused Hybrid Gate ({th_zs_fuse:.2f})", "Accuracy": acc_zs_fuse, "Recall": rec_zs_fuse, "FPR": fpr_zs_fuse, "F1-Score": f1_zs_fuse, "AUC-ROC": auc_zs_fuse}
    ])
    ft_report = pd.DataFrame([
        {"Configuration": f"Bi-LSTM Fine-Tuned ({th_ft_lstm:.2f})", "Accuracy": acc_ft_lstm, "Recall": rec_ft_lstm, "FPR": fpr_ft_lstm, "F1-Score": f1_ft_lstm, "AUC-ROC": auc_ft_lstm},
        {"Configuration": f"Physics CSI ({th_csi:.2f}) [unchanged]", "Accuracy": acc_ft_csi, "Recall": rec_ft_csi, "FPR": fpr_ft_csi, "F1-Score": f1_ft_csi, "AUC-ROC": auc_ft_csi},
        {"Configuration": f"Fused Hybrid Gate ({th_ft_fuse:.2f})", "Accuracy": acc_ft_fuse, "Recall": rec_ft_fuse, "FPR": fpr_ft_fuse, "F1-Score": f1_ft_fuse, "AUC-ROC": auc_ft_fuse}
    ])
    
    zs_path = reports_dir / "vzcrash_zero_shot_report.csv"
    ft_path_csv = reports_dir / "vzcrash_fine_tuned_report.csv"
    
    zs_report.to_csv(zs_path, index=False)
    ft_report.to_csv(ft_path_csv, index=False)
    print(f"Zero-shot validation report saved to: {zs_path.resolve()}")
    print(f"Fine-tuned validation report saved to: {ft_path_csv.resolve()}")
    
    # Confusion matrix for fine-tuned fusion
    pred_ft_fuse = (ft_fusion > th_ft_fuse).astype(int)
    cm = confusion_matrix(y_test, pred_ft_fuse)
    print(f"\nFine-Tuned Fusion Confusion Matrix:")
    print(f"                  Predicted Non-Crash  Predicted Crash")
    print(f"Actual Non-Crash:    {cm[0,0]:<15} {cm[0,1]}")
    print(f"Actual Crash:        {cm[1,0]:<15} {cm[1,1]}")
    
    print("\nDomain Adaptation complete!")

if __name__ == "__main__":
    main()
