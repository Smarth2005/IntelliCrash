import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config
from src.models.baselines import CNN1D, VanillaLSTM, LSTMAutoencoder
from src.models.train_bilstm import prepare_data

def run_baselines():
    print("Loading datasets via prepare_data()...")
    datasets = prepare_data()
    X_train_full = datasets["train"].X.numpy()
    y_train_full = datasets["train"].y_crash.numpy().flatten()
    X_test = datasets["test"].X.numpy()
    y_test = datasets["test"].y_crash.numpy().flatten()

    # Subset for faster training
    subset_size = 5000
    X_train = X_train_full[:subset_size]
    y_train = y_train_full[:subset_size]

    features_train = X_train[:, 0, 6:]
    features_test = X_test[:, 0, 6:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Baselines on {device}...")

    # 1. Static Thresholds
    y_pred_static = (features_test[:, 1] > 15.0).astype(int)
    acc_static = accuracy_score(y_test, y_pred_static)
    rec_static = recall_score(y_test, y_pred_static, zero_division=0)
    f1_static = f1_score(y_test, y_pred_static, zero_division=0)
    prec_static = precision_score(y_test, y_pred_static, zero_division=0)
    auc_static = 0.5  # Static threshold doesn't have true probabilities

    # 2. Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(features_train, y_train)
    y_pred_rf = rf.predict(features_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    rec_rf = recall_score(y_test, y_pred_rf, zero_division=0)
    f1_rf = f1_score(y_test, y_pred_rf, zero_division=0)
    from sklearn.metrics import roc_auc_score
    y_prob_rf = rf.predict_proba(features_test)[:, 1]
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    prec_rf = precision_score(y_test, y_pred_rf, zero_division=0)

    # PyTorch DataLoaders
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)), batch_size=256, shuffle=False)

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
        all_preds = []
        all_probs = []
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
        
        from sklearn.metrics import precision_score, roc_auc_score
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

    # 5. LSTM Autoencoder
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
    ae_preds = []
    all_mse = []
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
    rec_ae = recall_score(y_test, ae_preds, zero_division=0)
    f1_ae = f1_score(y_test, ae_preds, zero_division=0)
    prec_ae = precision_score(y_test, ae_preds, zero_division=0)
    
    # Normalizing MSE to 0-1 for a pseudo-AUC
    if np.max(all_mse) > 0:
        mse_norm = all_mse / np.max(all_mse)
    else:
        mse_norm = all_mse
        
    auc_ae = roc_auc_score(y_test, mse_norm)

    print("\nEvaluating Trained Bi-LSTM Model...")
    from src.models.bilstm import IntelliCrashBiLSTM
    cfg = get_config()
    bilstm = IntelliCrashBiLSTM(input_size=24, hidden_size=128, num_layers=2).to(device)
    checkpoint_path = Path(cfg["paths"]["model_checkpoints"]) / "best_bilstm.pth"
    if checkpoint_path.exists():
        bilstm.load_state_dict(torch.load(checkpoint_path, map_location=device))
        bilstm.eval()
        
        all_bilstm_preds = []
        all_bilstm_probs = []
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

    print("\n=========================================================================================================")
    print("                        TABLE: Architectural Baseline Comparison (Test Set Metrics)")
    print("=========================================================================================================")
    print(f"{'Model':<30} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1 Score':<8} | {'AUC-ROC':<8}")
    print("-" * 105)
    print(f"1. Static Thresholds            | {acc_static*100:>7.2f}% | {prec_static*100:>8.2f}% | {rec_static*100:>7.2f}% | {f1_static*100:>7.2f}% | {auc_static:>8.3f}")
    print(f"2. Random Forest (18 Features)  | {acc_rf*100:>7.2f}% | {prec_rf*100:>8.2f}% | {rec_rf*100:>7.2f}% | {f1_rf*100:>7.2f}% | {auc_rf:>8.3f}")
    print(f"3. 1D CNN                       | {acc_cnn*100:>7.2f}% | {prec_cnn*100:>8.2f}% | {rec_cnn*100:>7.2f}% | {f1_cnn*100:>7.2f}% | {auc_cnn:>8.3f}")
    print(f"4. Vanilla LSTM                 | {acc_vlstm*100:>7.2f}% | {prec_vlstm*100:>8.2f}% | {rec_vlstm*100:>7.2f}% | {f1_vlstm*100:>7.2f}% | {auc_vlstm:>8.3f}")
    print(f"5. Unsupervised LSTM Autoencoder| {acc_ae*100:>7.2f}% | {prec_ae*100:>8.2f}% | {rec_ae*100:>7.2f}% | {f1_ae*100:>7.2f}% | {auc_ae:>8.3f}")
    print(f"6. Bi-LSTM                      | {acc_bilstm*100:>7.2f}% | {prec_bilstm*100:>8.2f}% | {rec_bilstm*100:>7.2f}% | {f1_bilstm*100:>7.2f}% | {auc_bilstm:>8.3f}")
    print("=========================================================================================================")
    
    results_df = pd.DataFrame([
        {"Model": "1. Static Thresholds", "Accuracy": f"{acc_static*100:.2f}%", "Precision": f"{prec_static*100:.2f}%", "Recall": f"{rec_static*100:.2f}%", "F1 Score": f"{f1_static*100:.2f}%", "AUC-ROC": f"{auc_static:.3f}"},
        {"Model": "2. Random Forest (18 Features)", "Accuracy": f"{acc_rf*100:.2f}%", "Precision": f"{prec_rf*100:.2f}%", "Recall": f"{rec_rf*100:.2f}%", "F1 Score": f"{f1_rf*100:.2f}%", "AUC-ROC": f"{auc_rf:.3f}"},
        {"Model": "3. 1D CNN", "Accuracy": f"{acc_cnn*100:.2f}%", "Precision": f"{prec_cnn*100:.2f}%", "Recall": f"{rec_cnn*100:.2f}%", "F1 Score": f"{f1_cnn*100:.2f}%", "AUC-ROC": f"{auc_cnn:.3f}"},
        {"Model": "4. Vanilla LSTM", "Accuracy": f"{acc_vlstm*100:.2f}%", "Precision": f"{prec_vlstm*100:.2f}%", "Recall": f"{rec_vlstm*100:.2f}%", "F1 Score": f"{f1_vlstm*100:.2f}%", "AUC-ROC": f"{auc_vlstm:.3f}"},
        {"Model": "5. Unsup. LSTM Autoencoder", "Accuracy": f"{acc_ae*100:.2f}%", "Precision": f"{prec_ae*100:.2f}%", "Recall": f"{rec_ae*100:.2f}%", "F1 Score": f"{f1_ae*100:.2f}%", "AUC-ROC": f"{auc_ae:.3f}"},
        {"Model": "6. IntelliCrash (Bi-LSTM Gate)", "Accuracy": f"{acc_bilstm*100:.2f}%", "Precision": f"{prec_bilstm*100:.2f}%", "Recall": f"{rec_bilstm*100:.2f}%", "F1 Score": f"{f1_bilstm*100:.2f}%", "AUC-ROC": f"{auc_bilstm:.3f}"}
    ])
    
    cfg = get_config()
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "table_baselines.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n=> Table successfully exported to CSV for LaTeX insertion: {csv_path}")

    # Generate a pure-numeric CSV specifically for topsis.py
    topsis_df = pd.DataFrame([
        {"Model": "1. Static Thresholds", "Accuracy": acc_static, "Precision": prec_static, "Recall": rec_static, "F1 Score": f1_static, "AUC-ROC": auc_static},
        {"Model": "2. Random Forest", "Accuracy": acc_rf, "Precision": prec_rf, "Recall": rec_rf, "F1 Score": f1_rf, "AUC-ROC": auc_rf},
        {"Model": "3. 1D CNN", "Accuracy": acc_cnn, "Precision": prec_cnn, "Recall": rec_cnn, "F1 Score": f1_cnn, "AUC-ROC": auc_cnn},
        {"Model": "4. Vanilla LSTM", "Accuracy": acc_vlstm, "Precision": prec_vlstm, "Recall": rec_vlstm, "F1 Score": f1_vlstm, "AUC-ROC": auc_vlstm},
        {"Model": "5. Unsup. LSTM Autoencoder", "Accuracy": acc_ae, "Precision": prec_ae, "Recall": rec_ae, "F1 Score": f1_ae, "AUC-ROC": auc_ae},
        {"Model": "6. IntelliCrash (Bi-LSTM Gate)", "Accuracy": acc_bilstm, "Precision": prec_bilstm, "Recall": rec_bilstm, "F1 Score": f1_bilstm, "AUC-ROC": auc_bilstm}
    ])
    topsis_path = reports_dir / "topsis_input.csv"
    topsis_df.to_csv(topsis_path, index=False)
    print(f"=> Pure numeric table successfully exported for TOPSIS: {topsis_path}")

if __name__ == "__main__":
    run_baselines()
