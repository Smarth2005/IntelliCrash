"""
IntelliCrash — Rash Driving Classifier.

Trains a lightweight XGBoost model on the 18 engineered features to classify 
5 distinct rash driving patterns (plus normal and crash).
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config
from src.features.feature_engineering import FEATURE_NAMES

# Rash Classes:
# 0: Normal
# 1: Lane Weaving
# 2: Lane Swerving
# 3: Hard Braking
# 4: Hard Cornering
# 5: Quick U-turn
# 6: Crash
RASH_CLASS_NAMES = [
    "Normal", 
    "Lane Weaving", 
    "Lane Swerving", 
    "Hard Braking", 
    "Hard Cornering", 
    "Quick U-turn", 
    "Crash"
]

def load_data():
    """Load the engineered features and corresponding rash class labels."""
    cfg = get_config()
    processed_dir = Path(cfg["paths"]["processed_dir"])
    windows_dir = Path(cfg["paths"]["windows_dir"])
    
    # Load features (N, 18)
    print("Loading engineered features...")
    X = np.load(processed_dir / "imu_features.npy")
    
    # Load metadata to get rash labels
    print("Loading metadata...")
    meta_df = pd.read_parquet(windows_dir / "imu_metadata.parquet")
    
    # Ensure rash_class exists
    if "rash_class" not in meta_df.columns:
        raise ValueError("rash_class not found in metadata! Run preprocess_imu.py first.")
        
    y = meta_df["rash_class"].values
    
    print(f"Data loaded: X={X.shape}, y={y.shape}")
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for cls, count in zip(unique, counts):
        print(f"  {RASH_CLASS_NAMES[int(cls)]}: {count} ({count/len(y)*100:.2f}%)")
        
    return X, y

def train():
    """Train the XGBoost Rash Driving Classifier."""
    X, y = load_data()
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Encode labels to be sequential (0, 1, 2, ...)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_present_classes = len(le.classes_)

    print("\nTraining XGBoost Classifier...")
    # Using XGBoost for multi-class classification
    clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=num_present_classes,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,       # Slightly shallower than severity model to keep it fast
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    
    clf.fit(
        X_train, y_train_enc,
        eval_set=[(X_train, y_train_enc), (X_test, y_test_enc)],
        verbose=10
    )
    
    # Evaluation
    print("\nEvaluating Model...")
    y_pred_enc = clf.predict(X_test)
    y_pred = le.inverse_transform(y_pred_enc)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    # Filter class names to only those present in test set to avoid warnings
    present_classes = np.unique(y_test)
    target_names = [RASH_CLASS_NAMES[i] for i in present_classes]
    
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    
    # Save model and label encoder
    cfg = get_config()
    out_dir = Path(cfg["paths"]["model_checkpoints"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = out_dir / "xgboost_rash_classifier.json"
    clf.save_model(model_path)
    joblib.dump(le, out_dir / "rash_label_encoder.pkl")
    print(f"\nSaved trained model and label encoder to {out_dir}")
    
if __name__ == "__main__":
    train()
