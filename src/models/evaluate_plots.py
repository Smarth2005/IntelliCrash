"""
IntelliCrash — Model Evaluation & Plotting Script

Generates multi-class ROC curves, multi-class confusion matrices (7x7), 
binary ROC curves, and binary confusion matrices (2x2) for research paper figures.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import xgboost as xgb
import joblib

from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config
from src.models.bilstm import IntelliCrashBiLSTM
from src.features.feature_engineering import compute_csi

# Premium Academic Styling
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (10, 8),
    "figure.dpi": 300,
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
})

RASH_CLASS_NAMES = ["Normal", "Lane Weaving", "Lane Swerving", "Hard Braking", "Hard Cornering", "Quick U-turn"]

def ensure_plot_dir():
    cfg = get_config()
    out = Path(cfg["paths"]["plots_dir"]) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    return out

def load_test_data():
    cfg = get_config()
    data_dir = Path(cfg["paths"]["windows_dir"])
    X_test = np.load(data_dir / "imu_augmented_X_test.npy")
    y_test = np.load(data_dir / "imu_augmented_y_test.npy")
    # For xgboost, we need features. It's the last 18 channels of X.
    features_test = X_test[:, 0, 6:] 
    
    # Also load metadata to get true 6-class labels
    # Recreate the exact XGBoost test split
    processed_dir = Path(cfg["paths"]["processed_dir"])
    try:
        X_xgb = np.load(processed_dir / "imu_features.npy")
        meta_df = pd.read_parquet(data_dir / "imu_metadata.parquet")
        y_xgb = meta_df['rash_class'].values
        from sklearn.model_selection import train_test_split
        _, X_xgb_test, _, y_xgb_test = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb)
    except Exception as e:
        print(f"Error evaluating models: {e}. Make sure they are trained first.")
        X_xgb_test, y_xgb_test = None, None
        
    return X_test, y_test, X_xgb_test, y_xgb_test

def plot_multiclass_evaluation(X_xgb_test, y_xgb_test, out_dir):
    """Plot 6x6 Confusion Matrix and Multi-Class ROC for Best TOPSIS Model."""
    cfg = get_config()
    model_dir = Path(cfg["paths"]["model_checkpoints"])
    
    # Load model and encoder
    clf = joblib.load(model_dir / "best_rash_classifier.pkl")
    le = joblib.load(model_dir / "rash_label_encoder.pkl")
    
    y_pred_prob = clf.predict_proba(X_xgb_test)
    y_pred_enc = np.argmax(y_pred_prob, axis=1)
    y_test_enc = le.transform(y_xgb_test)
    
    # 1. 6x6 Confusion Matrix
    y_pred_actual = le.inverse_transform(y_pred_enc)
    cm = confusion_matrix(y_xgb_test, y_pred_actual, labels=range(6))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=RASH_CLASS_NAMES, yticklabels=RASH_CLASS_NAMES, 
                cbar_kws={'label': 'Count'})
    plt.title("Best Model (TOPSIS) 6-Class Confusion Matrix: Rash Driving Classification", pad=20)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "best_model_6x6_confusion_matrix.png", dpi=300)
    plt.close()
    
    # 2. Multi-Class ROC (One-vs-Rest)
    y_test_bin = label_binarize(y_xgb_test, classes=range(6))
    n_classes = len(le.classes_)
    
    plt.figure(figsize=(10, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{RASH_CLASS_NAMES[i]} (AUC = {roc_auc:.2f})')
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Best Model (TOPSIS) Multi-Class ROC Curve (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "best_model_multiclass_roc.png", dpi=300)
    plt.close()
    print("  Generated Multi-class Evaluation Plots (Best TOPSIS Model).")

def plot_binary_evaluation(X_test, y_test, out_dir):
    """Plot 2x2 Confusion Matrix and Binary ROC for Bi-LSTM & Fusion Gate."""
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = IntelliCrashBiLSTM(input_size=32, hidden_size=128, num_layers=2).to(device)
    model.load_state_dict(torch.load(Path(cfg["paths"]["model_checkpoints"]) / "best_bilstm.pth", map_location=device))
    model.eval()
    
    lstm_probs = []
    with torch.no_grad():
        for i in range(0, len(X_test), 256):
            batch = torch.FloatTensor(X_test[i:i+256]).to(device)
            probs, _ = model(batch)
            lstm_probs.extend(probs.cpu().numpy().flatten())
            
    lstm_probs = np.array(lstm_probs)
    
    # Calculate CSI
    features_test = X_test[:, 0, 6:]
    csi_scores = compute_csi(features_test, mode="real_car")
    
    # Load TOPSIS-optimized fusion weights (saved by optimize_weights.py)
    import json
    weights_path = Path(cfg["paths"]["reports_dir"]) / "optimal_fusion_weights.json"
    try:
        with open(weights_path) as f:
            opt = json.load(f)
        w_ml, w_csi = opt["w_ml"], opt["w_csi"]
        print(f"  Using TOPSIS-optimized weights: {w_ml} ML + {w_csi} CSI")
    except FileNotFoundError:
        w_ml, w_csi = 0.7, 0.3
        print(f"  WARNING: optimal_fusion_weights.json not found. Using default {w_ml}:{w_csi}")
    fusion_scores = (w_ml * lstm_probs) + (w_csi * csi_scores)
    y_pred_fusion = (fusion_scores > 0.5).astype(int)
    y_pred_lstm = (lstm_probs > 0.5).astype(int)
    
    # 1. 2x2 Confusion Matrix (Fusion Gate)
    cm = confusion_matrix(y_test, y_pred_fusion)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=["Non-Crash", "Crash"], yticklabels=["Non-Crash", "Crash"])
    plt.title(f"Hybrid Fusion Gate Confusion Matrix\n({w_ml} ML + {w_csi} CSI)", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel('True Status')
    plt.xlabel('Predicted Status')
    plt.tight_layout()
    plt.savefig(out_dir / "fusion_gate_2x2_cm.png", dpi=300)
    plt.close()
    
    # 2. Comparative ROC Curve
    plt.figure(figsize=(10, 8))
    
    # Fusion ROC
    fpr_f, tpr_f, _ = roc_curve(y_test, fusion_scores)
    auc_f = auc(fpr_f, tpr_f)
    plt.plot(fpr_f, tpr_f, color='red', lw=3, label=f'IntelliCrash Fusion Gate (AUC = {auc_f:.4f})')
    
    # Bi-LSTM ROC
    fpr_l, tpr_l, _ = roc_curve(y_test, lstm_probs)
    auc_l = auc(fpr_l, tpr_l)
    plt.plot(fpr_l, tpr_l, color='blue', lw=2, linestyle='--', label=f'Bi-LSTM Only (AUC = {auc_l:.4f})')
    
    # CSI Only ROC
    fpr_c, tpr_c, _ = roc_curve(y_test, csi_scores)
    auc_c = auc(fpr_c, tpr_c)
    plt.plot(fpr_c, tpr_c, color='green', lw=2, linestyle=':', label=f'Physics CSI Only (AUC = {auc_c:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparative ROC Curve: Crash Detection Architectures')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "comparative_binary_roc.png", dpi=300)
    plt.close()
    print("  Generated Binary Evaluation Plots (Bi-LSTM / Fusion).")

if __name__ == "__main__":
    out_dir = ensure_plot_dir()
    try:
        X_test, y_test, X_xgb_test, y_xgb_test = load_test_data()
        plot_multiclass_evaluation(X_xgb_test, y_xgb_test, out_dir)
        plot_binary_evaluation(X_test, y_test, out_dir)
        print(f"\nSuccessfully generated all evaluation plots in {out_dir}")
    except Exception as e:
        print(f"Error evaluating models: {e}. Make sure they are trained first.")
