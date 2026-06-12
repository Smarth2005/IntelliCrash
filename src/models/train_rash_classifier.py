"""
IntelliCrash — Rash Driving Classifier.

Trains a Stacking Ensemble on the 26 engineered features (18 physics + 8 temporal)
to classify 6 distinct rash driving patterns (crash excluded — handled by Bi-LSTM).
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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import make_pipeline

# Models for Stacking Ensemble
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config, set_all_seeds

# Set universal seed for 100% reproducible results across runs
set_all_seeds(42)

# Rash Classes:
# 0: Normal
# 1: Lane Weaving
# 2: Lane Swerving
# 3: Hard Braking
# 4: Hard Cornering
# 5: Quick U-turn
RASH_CLASS_NAMES = [
    "Normal", 
    "Lane Weaving", 
    "Lane Swerving", 
    "Hard Braking", 
    "Hard Cornering", 
    "Quick U-turn"
]

def load_data():
    cfg = get_config()
    data_dir = Path(cfg["paths"]["windows_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    
    # We use the physical features engineered from the raw data
    X = np.load(processed_dir / "imu_features.npy")
    
    # We load the 6-class labels from metadata (unaugmented)
    meta_df = pd.read_parquet(data_dir / "imu_metadata.parquet")
    y = meta_df['rash_class'].values
    
    # CRITICAL: Filter out crash class (6) — crash detection is handled
    # separately by the Bi-LSTM binary classifier. The rash driving
    # pattern classifier is strictly 6-class (0-5).
    non_crash_mask = y != 6
    X = X[non_crash_mask]
    y = y[non_crash_mask]
    
    crash_removed = (~non_crash_mask).sum()
    if crash_removed > 0:
        print(f"  Filtered out {crash_removed:,} crash windows (class 6)")
    print(f"  Final dataset: {len(X):,} samples, {len(np.unique(y))} classes")
    
    return X, y

def train():
    X, y = load_data()
    
    # 80/20 train/test split
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

    print("\nBuilding Stacking Ensemble Architecture...")
    # Base estimators for stacking
    estimators = [
        ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1))),
        ('mlp', make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))),
        ('gb', HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=5, random_state=42))
    ]
    
    # Stacking Classifier
    clf = StackingClassifier(
        estimators=estimators, 
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000)
    )
    
    print("\nTraining Stacking Ensemble (this may take a minute)...")
    clf.fit(X_train, y_train_enc)
    
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
    
    # Save as .pkl because StackingClassifier is a scikit-learn object
    model_path = out_dir / "stacking_rash_classifier.pkl"
    joblib.dump(clf, model_path)
    joblib.dump(le, out_dir / "rash_label_encoder.pkl")
    print(f"\nSaved trained Stacking Ensemble and label encoder to {out_dir}")
    
if __name__ == "__main__":
    train()
