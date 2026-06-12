"""
IntelliCrash — Architectural Baseline Comparison.

Trains and evaluates 6 models on the crash detection task:
1. Static Thresholds (Delta-V > 15 km/h)
2. Random Forest (18 engineered features)
3. 1D CNN (time-series)
4. Vanilla LSTM (time-series)
5. Unsupervised LSTM Autoencoder (anomaly detection)
6. IntelliCrash Bi-LSTM Fusion Gate (proposed)

Outputs:
- Console table
- outputs/reports/baseline_comparison.csv  (formatted)
- outputs/reports/topsis_input.csv         (pure numeric for TOPSIS)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config
from src.features.feature_engineering import extract_features_batch, compute_csi, NUM_ENGINEERED_FEATURES

# Total features = 6 raw sensor channels + N engineered features
TOTAL_FEATURES = 6 + NUM_ENGINEERED_FEATURES

# ── Baseline Model Definitions ───────────────────────────────────────────────

class CNN1D(nn.Module):
    def __init__(self, in_channels=TOTAL_FEATURES):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 99, 1) # 200 -> 198 -> 99
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return self.sigmoid(x)

class VanillaLSTM(nn.Module):
    def __init__(self, input_size=TOTAL_FEATURES):
        super(VanillaLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size=TOTAL_FEATURES):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, 32, batch_first=True)
        self.decoder = nn.LSTM(32, input_size, batch_first=True)
    def forward(self, x):
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded


# ── Data Loading (self-contained, no dependency on train_bilstm.py) ──────────

def load_test_data():
    """Load the saved test data and prepare train/test splits for baselines."""
    cfg = get_config()
    data_dir = Path(cfg["paths"]["windows_dir"])

    print("Loading augmented IMU dataset...")
    X_raw = np.load(data_dir / "imu_augmented_X.npy")
    y = np.load(data_dir / "imu_augmented_y.npy")

    print(f"Extracting features for {len(X_raw):,} windows...")
    features_1d = extract_features_batch(X_raw)

    # Combine raw (6) + engineered (N) → (batch, seq, 6+N)
    seq_len = X_raw.shape[1]
    features_expanded = np.repeat(features_1d[:, np.newaxis, :], seq_len, axis=1)
    X_full = np.concatenate([X_raw, features_expanded], axis=2)
    print(f"  Combined features: {X_full.shape[2]} per timestep (6 raw + {NUM_ENGINEERED_FEATURES} engineered)")

    # Split: 70% train, 15% val, 15% test (matching train_bilstm.py)
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_full, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, _, y_train, _ = train_test_split(
        X_temp, y_temp, test_size=0.15/(1-0.15), stratify=y_temp, random_state=42
    )

    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    return X_train, y_train, X_test, y_test


# ── Main ─────────────────────────────────────────────────────────────────────

def run_baselines():
    X_train_full, y_train_full, X_test, y_test = load_test_data()

    # Subset for faster training of shallow baselines
    subset_size = min(5000, len(X_train_full))
    X_train = X_train_full[:subset_size]
    y_train = y_train_full[:subset_size]

    # Tabular features (18 engineered, from first timestep since they're broadcast)
    features_train = X_train[:, 0, 6:]
    features_test = X_test[:, 0, 6:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training baselines on {device}...\n")

    # 1. Static Thresholds (Delta-V > 15 km/h)
    y_pred_static = (features_test[:, 1] > 15.0).astype(int)
    acc_static = accuracy_score(y_test, y_pred_static)
    prec_static = precision_score(y_test, y_pred_static, zero_division=0)
    rec_static = recall_score(y_test, y_pred_static, zero_division=0)
    f1_static = f1_score(y_test, y_pred_static, zero_division=0)
    auc_static = 0.5  # No probabilistic output

    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(features_train, y_train)
    y_pred_rf = rf.predict(features_test)
    y_prob_rf = rf.predict_proba(features_test)[:, 1]
    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf, zero_division=0)
    rec_rf = recall_score(y_test, y_pred_rf, zero_division=0)
    f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    # PyTorch DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=64, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=256, shuffle=False
    )

    def train_pytorch_model(model, epochs=3):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch).view(-1)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
        return model

    def evaluate_pytorch_model(model):
        model.eval()
        all_preds, all_probs = [], []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch).view(-1)
                probs = outputs.cpu().numpy()
                if probs.ndim == 0: probs = [probs]
                all_probs.extend(probs)
                preds = (outputs > 0.5).int().cpu().numpy()
                if preds.ndim == 0: preds = [preds]
                all_preds.extend(preds)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        return (accuracy_score(y_test, all_preds),
                precision_score(y_test, all_preds, zero_division=0),
                recall_score(y_test, all_preds, zero_division=0),
                f1_score(y_test, all_preds, zero_division=0),
                roc_auc_score(y_test, all_probs))

    # 3. 1D CNN
    print("Training 1D CNN...")
    cnn_model = CNN1D().to(device)
    train_pytorch_model(cnn_model, epochs=3)
    acc_cnn, prec_cnn, rec_cnn, f1_cnn, auc_cnn = evaluate_pytorch_model(cnn_model)

    # 4. Vanilla LSTM
    print("Training Vanilla LSTM...")
    vanilla_lstm = VanillaLSTM().to(device)
    train_pytorch_model(vanilla_lstm, epochs=3)
    acc_vlstm, prec_vlstm, rec_vlstm, f1_vlstm, auc_vlstm = evaluate_pytorch_model(vanilla_lstm)

    # 5. LSTM Autoencoder (unsupervised anomaly detection)
    print("Training LSTM Autoencoder...")
    autoencoder = LSTMAutoencoder().to(device)
    normal_X = X_train[y_train == 0]
    ae_loader = DataLoader(TensorDataset(torch.FloatTensor(normal_X)), batch_size=64, shuffle=True)
    ae_opt = optim.Adam(autoencoder.parameters(), lr=0.001)
    autoencoder.train()
    for epoch in range(3):
        for (X_batch,) in ae_loader:
            X_batch = X_batch.to(device)
            ae_opt.zero_grad()
            recon = autoencoder(X_batch)
            loss = nn.MSELoss()(recon, X_batch)
            loss.backward()
            ae_opt.step()

    autoencoder.eval()
    ae_preds, all_mse = [], []
    with torch.no_grad():
        for (X_batch, _) in test_loader:
            X_batch = X_batch.to(device)
            recon = autoencoder(X_batch)
            mse_batch = torch.mean((recon - X_batch)**2, dim=[1,2])
            preds = (mse_batch > 0.5).int().cpu().numpy()
            ae_preds.extend(preds)
            all_mse.extend(mse_batch.cpu().numpy())

    ae_preds = np.array(ae_preds)
    all_mse = np.array(all_mse)
    acc_ae = accuracy_score(y_test, ae_preds)
    prec_ae = precision_score(y_test, ae_preds, zero_division=0)
    rec_ae = recall_score(y_test, ae_preds, zero_division=0)
    f1_ae = f1_score(y_test, ae_preds, zero_division=0)
    mse_norm = all_mse / np.max(all_mse) if np.max(all_mse) > 0 else all_mse
    auc_ae = roc_auc_score(y_test, mse_norm)

    # 6. IntelliCrash Bi-LSTM (proposed — load trained checkpoint)
    print("\nEvaluating trained Bi-LSTM (IntelliCrash)...")
    from src.models.bilstm import IntelliCrashBiLSTM
    cfg = get_config()
    bilstm = IntelliCrashBiLSTM(input_size=TOTAL_FEATURES, hidden_size=128, num_layers=2).to(device)
    checkpoint_path = Path(cfg["paths"]["model_checkpoints"]) / "best_bilstm.pth"
    if checkpoint_path.exists():
        bilstm.load_state_dict(torch.load(checkpoint_path, map_location=device))
        bilstm.eval()
        all_bilstm_preds, all_bilstm_probs = [], []
        with torch.no_grad():
            for X_batch, _ in test_loader:
                X_batch = X_batch.to(device)
                crash_prob, _ = bilstm(X_batch)
                probs = crash_prob.view(-1).cpu().numpy()
                all_bilstm_probs.extend(probs)
                preds = (probs > 0.5).astype(int)
                all_bilstm_preds.extend(preds)
        acc_bilstm = accuracy_score(y_test, all_bilstm_preds)
        prec_bilstm = precision_score(y_test, all_bilstm_preds, zero_division=0)
        rec_bilstm = recall_score(y_test, all_bilstm_preds, zero_division=0)
        f1_bilstm = f1_score(y_test, all_bilstm_preds, zero_division=0)
        auc_bilstm = roc_auc_score(y_test, all_bilstm_probs)
    else:
        print("Warning: Bi-LSTM checkpoint not found. Using zeros.")
        acc_bilstm = prec_bilstm = rec_bilstm = f1_bilstm = auc_bilstm = 0.0

    # ── Print Results ────────────────────────────────────────────────────────
    print("\n=========================================================================================================")
    print("                        Architectural Baseline Comparison (Test Set Metrics)")
    print("=========================================================================================================")
    print(f"{'Model':<30} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1 Score':<8} | {'AUC-ROC':<8}")
    print("-" * 105)
    print(f"Static Thresholds              | {acc_static*100:>7.2f}% | {prec_static*100:>8.2f}% | {rec_static*100:>7.2f}% | {f1_static*100:>7.2f}% | {auc_static:>8.3f}")
    print(f"Random Forest (18 Features)    | {acc_rf*100:>7.2f}% | {prec_rf*100:>8.2f}% | {rec_rf*100:>7.2f}% | {f1_rf*100:>7.2f}% | {auc_rf:>8.3f}")
    print(f"1D CNN                         | {acc_cnn*100:>7.2f}% | {prec_cnn*100:>8.2f}% | {rec_cnn*100:>7.2f}% | {f1_cnn*100:>7.2f}% | {auc_cnn:>8.3f}")
    print(f"Vanilla LSTM                   | {acc_vlstm*100:>7.2f}% | {prec_vlstm*100:>8.2f}% | {rec_vlstm*100:>7.2f}% | {f1_vlstm*100:>7.2f}% | {auc_vlstm:>8.3f}")
    print(f"Unsup. LSTM Autoencoder        | {acc_ae*100:>7.2f}% | {prec_ae*100:>8.2f}% | {rec_ae*100:>7.2f}% | {f1_ae*100:>7.2f}% | {auc_ae:>8.3f}")
    print(f"IntelliCrash (Bi-LSTM Gate)    | {acc_bilstm*100:>7.2f}% | {prec_bilstm*100:>8.2f}% | {rec_bilstm*100:>7.2f}% | {f1_bilstm*100:>7.2f}% | {auc_bilstm:>8.3f}")
    print("=========================================================================================================")

    # ── Save CSVs ────────────────────────────────────────────────────────────
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Formatted CSV
    results_df = pd.DataFrame([
        {"Model": "Static Thresholds", "Accuracy": f"{acc_static*100:.2f}%", "Precision": f"{prec_static*100:.2f}%", "Recall": f"{rec_static*100:.2f}%", "F1 Score": f"{f1_static*100:.2f}%", "AUC-ROC": f"{auc_static:.3f}"},
        {"Model": "Random Forest (18 Features)", "Accuracy": f"{acc_rf*100:.2f}%", "Precision": f"{prec_rf*100:.2f}%", "Recall": f"{rec_rf*100:.2f}%", "F1 Score": f"{f1_rf*100:.2f}%", "AUC-ROC": f"{auc_rf:.3f}"},
        {"Model": "1D CNN", "Accuracy": f"{acc_cnn*100:.2f}%", "Precision": f"{prec_cnn*100:.2f}%", "Recall": f"{rec_cnn*100:.2f}%", "F1 Score": f"{f1_cnn*100:.2f}%", "AUC-ROC": f"{auc_cnn:.3f}"},
        {"Model": "Vanilla LSTM", "Accuracy": f"{acc_vlstm*100:.2f}%", "Precision": f"{prec_vlstm*100:.2f}%", "Recall": f"{rec_vlstm*100:.2f}%", "F1 Score": f"{f1_vlstm*100:.2f}%", "AUC-ROC": f"{auc_vlstm:.3f}"},
        {"Model": "Unsupervised LSTM Autoencoder", "Accuracy": f"{acc_ae*100:.2f}%", "Precision": f"{prec_ae*100:.2f}%", "Recall": f"{rec_ae*100:.2f}%", "F1 Score": f"{f1_ae*100:.2f}%", "AUC-ROC": f"{auc_ae:.3f}"},
        {"Model": "IntelliCrash (Bi-LSTM Gate)", "Accuracy": f"{acc_bilstm*100:.2f}%", "Precision": f"{prec_bilstm*100:.2f}%", "Recall": f"{rec_bilstm*100:.2f}%", "F1 Score": f"{f1_bilstm*100:.2f}%", "AUC-ROC": f"{auc_bilstm:.3f}"},
    ])
    results_df.to_csv(reports_dir / "baseline_comparison.csv", index=False)
    print(f"\n=> Table saved to: {reports_dir / 'baseline_comparison.csv'}")

    # Pure-numeric CSV for TOPSIS
    topsis_df = pd.DataFrame([
        {"Model": "Static Thresholds", "Accuracy": acc_static, "Precision": prec_static, "Recall": rec_static, "F1 Score": f1_static, "AUC-ROC": auc_static},
        {"Model": "Random Forest", "Accuracy": acc_rf, "Precision": prec_rf, "Recall": rec_rf, "F1 Score": f1_rf, "AUC-ROC": auc_rf},
        {"Model": "1D CNN", "Accuracy": acc_cnn, "Precision": prec_cnn, "Recall": rec_cnn, "F1 Score": f1_cnn, "AUC-ROC": auc_cnn},
        {"Model": "Vanilla LSTM", "Accuracy": acc_vlstm, "Precision": prec_vlstm, "Recall": rec_vlstm, "F1 Score": f1_vlstm, "AUC-ROC": auc_vlstm},
        {"Model": "Unsupervised LSTM Autoencoder", "Accuracy": acc_ae, "Precision": prec_ae, "Recall": rec_ae, "F1 Score": f1_ae, "AUC-ROC": auc_ae},
        {"Model": "IntelliCrash (Bi-LSTM Gate)", "Accuracy": acc_bilstm, "Precision": prec_bilstm, "Recall": rec_bilstm, "F1 Score": f1_bilstm, "AUC-ROC": auc_bilstm},
    ])
    topsis_df.to_csv(reports_dir / "topsis_input.csv", index=False)
    print(f"=> TOPSIS input saved to: {reports_dir / 'topsis_input.csv'}")

if __name__ == "__main__":
    run_baselines()
