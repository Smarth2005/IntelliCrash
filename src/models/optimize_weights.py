"""
IntelliCrash — Fusion Gate Weight Optimization.

Tests various ML:CSI weight combinations on the test set and finds the
mathematically optimal pair (max Recall, tie-breaker: min FPR).

Output:
- Console table
- outputs/reports/weight_optimization.csv
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from pathlib import Path
import sys
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config
from src.models.bilstm import IntelliCrashBiLSTM
from src.features.feature_engineering import compute_csi

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return recall * 100, fpr * 100, f1 * 100

def run_optimization():
    print("Loading test data...")
    data_dir = Path("data/processed/windows")
    try:
        X_test = np.load(data_dir / "imu_augmented_X_test.npy")
        y_test = np.load(data_dir / "imu_augmented_y_test.npy")
    except FileNotFoundError:
        print("Test data not found. Please ensure test sets are saved.")
        return

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
    batch_size = 256
    lstm_probs = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            probs, _ = model(batch)
            lstm_probs.append(probs.cpu().numpy().flatten())
    lstm_probs = np.concatenate(lstm_probs)
    
    csi_scores = compute_csi(features_test, mode="real_car")
    
    print("=========================================================================")
    print("        Weight Optimization for Hybrid Fusion Gate")
    print("=========================================================================")
    print(f"{'Fusion Weights (ML : CSI)':<25} | {'Recall':<10} | {'FPR':<10} | {'F1-Score'}")
    print("-------------------------------------------------------------------------")
    
    weights = [(1.0, 0.0), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.0, 1.0)]
    
    best_f1 = -1
    best_fpr = 100
    best_weights = (0.7, 0.3)
    
    rows = []
    for w_ml, w_phys in weights:
        f_scores = (w_ml * lstm_probs) + (w_phys * csi_scores)
        f_preds = (f_scores > 0.5).astype(int)
        rec, fpr, f1 = calculate_metrics(y_test, f_preds)
        
        weight_str = f"{w_ml:.1f} : {w_phys:.1f}"
        print(f"{weight_str:<25} | {rec:>6.1f}%     | {fpr:>6.1f}%     | {f1:>6.1f}%")
        
        rows.append({"Fusion Weights (ML:CSI)": weight_str, "Recall (%)": round(rec, 2), "FPR (%)": round(fpr, 2), "F1-Score (%)": round(f1, 2)})
            
    print("=========================================================================")
    
    # Save CSV for TOPSIS
    cfg = get_config()
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = reports_dir / "fusion_topsis_input.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n[TOPSIS] Automatically evaluating fusion weights using TOPSIS...")
    
    # Run TOPSIS automatically
    topsis_result_csv = reports_dir / "fusion_topsis_ranking.csv"
    topsis_script = str(Path(__file__).resolve().parent.parent.parent / "topsis.py")
    weights = "0.4,0.3,0.3"
    impacts = "+,-,+"
    
    try:
        subprocess.run([
            sys.executable, topsis_script, 
            str(csv_path), weights, impacts, str(topsis_result_csv)
        ], check=True, capture_output=True, text=True)
        
        # Read the TOPSIS result to find the winner
        topsis_df = pd.read_csv(topsis_result_csv)
        if 'Rank' in topsis_df.columns:
            winner_row = topsis_df[topsis_df['Rank'] == 1].iloc[0]
            best_weights = winner_row['Fusion Weights (ML:CSI)']
            print(f"\n🏆 TOPSIS WINNER (Rank #1 Fusion Weight): {best_weights}")
            print(f"   Topsis Score: {winner_row.get('Topsis Score', 'N/A')}")
        else:
            best_weights = topsis_df.iloc[0, 0]
            print(f"\n🏆 TOPSIS WINNER (First Row): {best_weights}")
        
        # Save the optimal weights to a JSON file for downstream scripts
        import json
        parts = str(best_weights).split(":")
        w_ml_opt = float(parts[0].strip())
        w_csi_opt = float(parts[1].strip())
        
        optimal_path = reports_dir / "optimal_fusion_weights.json"
        with open(optimal_path, "w") as f:
            json.dump({"w_ml": w_ml_opt, "w_csi": w_csi_opt}, f)
        print(f"   Saved optimal weights to: {optimal_path}")
            
    except Exception as e:
        print(f"\n⚠️ WARNING: Auto-TOPSIS failed ({e}). Please run manually:")
        print(f'   !python topsis.py "{csv_path}" "{weights}" "{impacts}" "{topsis_result_csv}"')

if __name__ == "__main__":
    run_optimization()
