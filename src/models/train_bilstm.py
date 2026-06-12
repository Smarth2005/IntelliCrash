"""
IntelliCrash — LSTM Training Script.

Trains the Bi-LSTM crash detection model on the augmented dataset.
Saves model checkpoints and exports to ONNX for edge deployment.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config, set_all_seeds
from src.models.bilstm import IntelliCrashBiLSTM, IMUDataset, FocalLoss
from src.features.feature_engineering import extract_features_batch, compute_csi

# Set universal seed for 100% reproducible results across runs
set_all_seeds(42)

def prepare_data(val_split=0.15, test_split=0.15, seed=42):
    """Load augmented windows and prepare PyTorch datasets."""
    cfg = get_config()
    data_dir = Path(cfg["paths"]["windows_dir"])
    
    # Load augmented data
    print("Loading augmented IMU dataset...")
    X_raw = np.load(data_dir / "imu_augmented_X.npy")
    y = np.load(data_dir / "imu_augmented_y.npy")
    
    print(f"Extracting features for {len(X_raw):,} windows (this may take a minute)...")
    # Features shape: (num_windows, 18)
    features_1d = extract_features_batch(X_raw)
    
    # Compute CSI severity scores
    severity = compute_csi(features_1d, mode="real_car")
    
    # Combine raw (6) and engineered (18) into (batch, seq, 24)
    # Broadcast features to match sequence length
    seq_len = X_raw.shape[1]
    features_expanded = np.repeat(features_1d[:, np.newaxis, :], seq_len, axis=1)
    
    # Combine
    X_full = np.concatenate([X_raw, features_expanded], axis=2)
    print(f"Final input shape: {X_full.shape}")
    
    # Splitting
    test_frac = test_split
    val_frac = val_split / (1 - test_frac)
    
    X_temp, X_test, y_temp, y_test, sev_temp, sev_test = train_test_split(
        X_full, y, severity, test_size=test_frac, stratify=y, random_state=seed
    )
    
    X_train, X_val, y_train, y_val, sev_train, sev_val = train_test_split(
        X_temp, y_temp, sev_temp, test_size=val_frac, stratify=y_temp, random_state=seed
    )
    
    print(f"Splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Save test sets for evaluation script
    np.save(data_dir / "imu_augmented_X_test.npy", X_test)
    np.save(data_dir / "imu_augmented_y_test.npy", y_test)
    
    return {
        "train": IMUDataset(X_train, y_train, sev_train),
        "val": IMUDataset(X_val, y_val, sev_val),
        "test": IMUDataset(X_test, y_test, sev_test)
    }


def train_model():
    cfg = get_config()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparams
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    lr = cfg["training"]["learning_rate"]
    
    # Data
    datasets = prepare_data()
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=batch_size)
    
    # Model
    model = IntelliCrashBiLSTM(
        input_size=32, 
        hidden_size=cfg["lstm"]["hidden_size"],
        num_layers=cfg["lstm"]["num_layers"],
        dropout=cfg["lstm"]["dropout"]
    ).to(device)
    
    # Losses
    criterion_crash = FocalLoss(alpha=cfg["training"]["focal_loss_alpha"], 
                                gamma=cfg["training"]["focal_loss_gamma"])
    criterion_sev = torch.nn.MSELoss()
    
    w_crash = cfg["training"]["crash_loss_weight"]
    w_sev = cfg["training"]["severity_loss_weight"]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    out_dir = Path(cfg["paths"]["model_checkpoints"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_crash_acc = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            X = batch['features'].to(device)
            y_crash = batch['crash'].to(device)
            y_sev = batch['severity'].to(device)
            
            optimizer.zero_grad()
            pred_crash, pred_sev = model(X)
            
            loss_crash = criterion_crash(pred_crash, y_crash)
            loss_sev = criterion_sev(pred_sev, y_sev)
            loss = w_crash * loss_crash + w_sev * loss_sev
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X.size(0)
            train_crash_acc += ((pred_crash > 0.5) == y_crash).sum().item()
            
        scheduler.step()
        
        train_loss /= len(datasets["train"])
        train_crash_acc /= len(datasets["train"])
        
        # Eval
        model.eval()
        val_loss = 0.0
        val_crash_acc = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                X = batch['features'].to(device)
                y_crash = batch['crash'].to(device)
                y_sev = batch['severity'].to(device)
                
                pred_crash, pred_sev = model(X)
                
                loss_crash = criterion_crash(pred_crash, y_crash)
                loss_sev = criterion_sev(pred_sev, y_sev)
                loss = w_crash * loss_crash + w_sev * loss_sev
                
                val_loss += loss.item() * X.size(0)
                val_crash_acc += ((pred_crash > 0.5) == y_crash).sum().item()
                
        val_loss /= len(datasets["val"])
        val_crash_acc /= len(datasets["val"])
        
        print(f"Epoch {epoch+1:02d}: Train Loss={train_loss:.4f} Acc={train_crash_acc:.4f} | "
              f"Val Loss={val_loss:.4f} Acc={val_crash_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), out_dir / "best_bilstm.pth")
            print(f"  -> Saved new best model")
            
    print("Training complete!")
    return model, datasets["test"]


def export_to_onnx(model, dummy_input_shape=(1, 200, 32)):
    cfg = get_config()
    out_dir = Path(cfg["paths"]["model_onnx"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = next(model.parameters()).device
    dummy_input = torch.randn(dummy_input_shape).to(device)
    
    onnx_path = out_dir / "intellicrash_lstm.onnx"
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['crash_prob', 'severity'],
        dynamic_axes={'input': {0: 'batch_size'},
                      'crash_prob': {0: 'batch_size'},
                      'severity': {0: 'batch_size'}}
    )
    print(f"\nExported ONNX model to {onnx_path}")


if __name__ == "__main__":
    # Lower epochs for quick local test if requested
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    
    if args.epochs is not None:
        cfg = get_config()
        cfg["training"]["epochs"] = args.epochs
        
    trained_model, test_dataset = train_model()
    
    # Load best weights before ONNX export
    trained_model.load_state_dict(torch.load(Path(get_config()["paths"]["model_checkpoints"]) / "best_bilstm.pth"))
    export_to_onnx(trained_model)
