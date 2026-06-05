import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score
from pathlib import Path
import sys

# Ensure imports work when running from command line
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.models.lstm import IntelliCrashLSTM
from src.features.feature_engineering import compute_csi

def calculate_metrics(y_true, y_pred):
    """Calculate Recall and False Positive Rate (FPR)"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return recall * 100, fpr * 100

def run_ablation_study():
    print("Loading test data...")
    # In Colab, you would load the actual X_test and y_test saved during preprocessing.
    # We will assume they are saved in processed_dir/
    data_dir = Path("data/processed")
    
    try:
        X_test = np.load(data_dir / "imu_augmented_X_test.npy")
        y_test = np.load(data_dir / "imu_augmented_y_test.npy")
    except FileNotFoundError:
        print("Test data not found. Please ensure preprocess_imu.py has been run and test sets are saved.")
        return

    # Extract the 18 engineered features (they are the last 18 columns in the 24-channel input)
    # We just need the features from the first timestep since they are broadcasted
    features_test = X_test[:, 0, 6:] 

    print("Loading trained Bi-LSTM model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntelliCrashLSTM(input_size=24, hidden_size=128, num_layers=2).to(device)
    
    try:
        model.load_state_dict(torch.load("models/checkpoints/best_lstm.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        print("Model checkpoint not found. Please train the LSTM first.")
        return

    print("Running Inference on Test Set...\n")
    
    # 1. Get LSTM Predictions
    X_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        lstm_probs, _ = model(X_tensor)
        lstm_probs = lstm_probs.cpu().numpy().flatten()
    
    # 2. Get Physics CSI Predictions
    csi_scores = compute_csi(features_test, mode="real_car")
    
    # 3. Calculate Fusion Scores
    fusion_scores = (0.6 * lstm_probs) + (0.4 * csi_scores)

    # Thresholding (0.5 threshold)
    pred_lstm = (lstm_probs > 0.5).astype(int)
    pred_physics = (csi_scores > 0.5).astype(int)
    pred_fusion = (fusion_scores > 0.5).astype(int)

    # Calculate Metrics
    recall_lstm, fpr_lstm = calculate_metrics(y_test, pred_lstm)
    recall_phys, fpr_phys = calculate_metrics(y_test, pred_physics)
    recall_fuse, fpr_fuse = calculate_metrics(y_test, pred_fusion)

    # Print Results Table
    print("-" * 65)
    print(f"{'Architecture Configuration':<30} | {'Recall':<12} | {'FPR':<12}")
    print("-" * 65)
    print(f"{'1. Physics CSI Gate Only':<30} | {recall_phys:>6.1f}% | {fpr_phys:>6.1f}%")
    print(f"{'2. Bi-LSTM Only':<30} | {recall_lstm:>6.1f}% | {fpr_lstm:>6.1f}%")
    print(f"{'3. Hybrid Fusion Gate':<30} | {recall_fuse:>6.1f}% | {fpr_fuse:>6.1f}%")
    print("-" * 65)
    print("\nCopy these results directly into your project_evaluation_report.md!")

if __name__ == "__main__":
    run_ablation_study()
