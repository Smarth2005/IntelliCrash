"""
IntelliCrash — Multi-Class Maneuver Baseline Comparison.

Trains and evaluates 11 standard ML models on the 6-class rash driving
pattern classification task (crash class excluded — handled by Bi-LSTM):
  1. XGBoost
  2. XGBoost (Tuned via RandomizedSearchCV)
  3. Random Forest
  4. Extra Trees
  5. Gradient Boosting (HistGradientBoosting)
  6. LightGBM
  7. CatBoost
  8. MLP (Neural Network)
  9. SVC (SVM)
 10. K-Nearest Neighbors (KNN)
 11. Stacking Ensemble
 12. Soft Voting Ensemble (top models)

Applies TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
to mathematically rank the best software model based on:
- Accuracy (30%)
- Macro Precision (15%)
- Macro Recall (15%)
- Macro F1-Score (40%)
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import joblib
import subprocess

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import make_pipeline

# Models
import xgboost as xgb
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Additional gradient boosting libraries
import lightgbm as lgb
from catboost import CatBoostClassifier

from scipy.stats import randint, uniform

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config, set_all_seeds

# Set seed for 100% reproducibility
set_all_seeds(42)

# 6-class rash driving labels (NO crash — crash is handled by Bi-LSTM separately)
RASH_CLASS_NAMES = [
    "Normal",           # 0
    "Lane Weaving",     # 1
    "Lane Swerving",    # 2
    "Hard Braking",     # 3
    "Hard Cornering",   # 4
    "Quick U-turn",     # 5
]

def load_data():
    cfg = get_config()
    data_dir = Path(cfg["paths"]["windows_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    
    print("Loading engineered features and metadata...")
    X = np.load(processed_dir / "imu_features.npy")
    meta_df = pd.read_parquet(data_dir / "imu_metadata.parquet")
    y = meta_df['rash_class'].values
    
    # ── CRITICAL: Filter out crash class (6) ──
    # Crash detection is handled by the Bi-LSTM binary classifier.
    # The rash driving pattern classifier is strictly 6-class (0-5).
    non_crash_mask = y != 6
    X = X[non_crash_mask]
    y = y[non_crash_mask]
    
    crash_removed = (~non_crash_mask).sum()
    if crash_removed > 0:
        print(f"  Filtered out {crash_removed:,} crash windows (class 6)")
    
    print(f"  Final dataset: {len(X):,} samples, {len(np.unique(y))} classes")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    return X, y

def main():
    X, y = load_data()
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    num_classes = len(le.classes_)
    print(f"\n  Training: {len(X_train):,} samples | Testing: {len(X_test):,} samples")
    print(f"  Number of classes: {num_classes}")
    
    # Compute class weights for imbalanced training
    sample_weights = compute_sample_weight("balanced", y_train_enc)
    
    # ── XGBoost Hyperparameter Tuning via RandomizedSearchCV ──
    print("\n[1/12] Running XGBoost Hyperparameter Tuning (RandomizedSearchCV)...")
    xgb_param_dist = {
        'n_estimators': randint(200, 1000),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.2),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5),
        'reg_alpha': uniform(0, 1.0),
        'reg_lambda': uniform(0.5, 2.0),
    }
    
    xgb_base = xgb.XGBClassifier(
        objective='multi:softprob', num_class=num_classes,
        eval_metric='mlogloss', n_jobs=-1, random_state=42,
        tree_method='hist',
    )
    
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_param_dist, n_iter=50, cv=3,
        scoring='f1_macro', random_state=42, n_jobs=-1, verbose=0
    )
    xgb_search.fit(X_train, y_train_enc, sample_weight=sample_weights)
    best_xgb_params = xgb_search.best_params_
    print(f"  Best XGBoost params: {best_xgb_params}")
    print(f"  Best CV F1 (macro): {xgb_search.best_score_:.4f}")
    
    # Base estimators for stacking
    estimators = [
        ('knn', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1))),
        ('mlp', make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42))),
        ('gb', HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=6, random_state=42)),
        ('lgbm', lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1)),
    ]
    
    # Dictionary of all models to evaluate
    models = {
        "XGBoost (Default)": xgb.XGBClassifier(
            objective='multi:softprob', num_class=num_classes,
            n_estimators=150, learning_rate=0.05, max_depth=6, 
            subsample=0.8, colsample_bytree=0.8, random_state=42, 
            eval_metric='mlogloss', n_jobs=-1
        ),
        "XGBoost (Tuned)": xgb.XGBClassifier(
            objective='multi:softprob', num_class=num_classes,
            eval_metric='mlogloss', n_jobs=-1, random_state=42,
            tree_method='hist',
            **best_xgb_params
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight='balanced', 
            random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=12, class_weight='balanced', 
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_depth=6, 
            random_state=42
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=8,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, verbose=-1,
            n_jobs=-1, min_child_samples=10
        ),
        "CatBoost": CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=8,
            auto_class_weights='Balanced', random_seed=42,
            verbose=0, thread_count=-1
        ),
        "MLP (Neural Network)": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, 
                          random_state=42, early_stopping=True,
                          validation_fraction=0.15, learning_rate='adaptive')
        ),
        "SVC (SVM)": make_pipeline(
            StandardScaler(),
            SVC(kernel='rbf', C=10.0, class_weight='balanced', 
                random_state=42, probability=False)
        ),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=7, weights='distance', 
                                 metric='minkowski', p=2, n_jobs=-1)
        ),
        "Stacking Ensemble": StackingClassifier(
            estimators=estimators, 
            final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
            cv=3, n_jobs=-1
        ),
        "Soft Voting Ensemble": VotingClassifier(
            estimators=[
                ('lgbm', lgb.LGBMClassifier(
                    n_estimators=300, learning_rate=0.05, max_depth=8,
                    class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
                )),
                ('knn', make_pipeline(
                    StandardScaler(),
                    KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1)
                )),
                ('et', ExtraTreesClassifier(
                    n_estimators=300, max_depth=12, class_weight='balanced',
                    random_state=42, n_jobs=-1
                )),
                ('catboost', CatBoostClassifier(
                    iterations=300, learning_rate=0.05, depth=8,
                    auto_class_weights='Balanced', random_seed=42, verbose=0
                )),
            ],
            voting='soft', n_jobs=-1
        ),
    }

    results = []
    total = len(models)
    for idx, (name, model) in enumerate(models.items(), 1):
        print(f"\n[{idx}/{total}] Training {name}...")
        
        # Train: Handle models that don't support sample weights natively
        no_sample_weight_models = [
            "MLP (Neural Network)", "K-Nearest Neighbors", "SVC (SVM)", 
            "Stacking Ensemble", "Soft Voting Ensemble"
        ]
        if name in no_sample_weight_models:
            model.fit(X_train, y_train_enc)
        else:
            model.fit(X_train, y_train_enc, sample_weight=sample_weights)
            
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test_enc, y_pred)
        prec = precision_score(y_test_enc, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test_enc, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test_enc, y_pred, average='macro', zero_division=0)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Macro Precision": prec,
            "Macro Recall": rec,
            "Macro F1-Score": f1
        })
        print(f"  => Accuracy: {acc:.4f} | F1 (macro): {f1:.4f}")

    results_df = pd.DataFrame(results)
    
    # Save to reports for external topsis.py tool
    cfg = get_config()
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = reports_dir / "multiclass_topsis_input.csv"
    results_df.to_csv(csv_path, index=False)
    
    # ── RUN TOPSIS.PY AUTOMATICALLY ──
    print("\n[TOPSIS] Running mathematical evaluation to find the optimal model...")
    topsis_result_csv = reports_dir / "multiclass_topsis_result.csv"
    topsis_script = str(Path(__file__).resolve().parent.parent.parent / "topsis.py")
    
    # Weights: Accuracy (0.30), Precision (0.15), Recall (0.15), F1 (0.40)
    weights = "0.30,0.15,0.15,0.40"
    impacts = "+,+,+,+"
    
    try:
        subprocess.run([
            sys.executable, topsis_script, 
            str(csv_path), weights, impacts, str(topsis_result_csv)
        ], check=True, capture_output=True, text=True)
        
        # Read the TOPSIS result
        topsis_df = pd.read_csv(topsis_result_csv)
        
        # Format for display
        display_df = topsis_df.copy()
        for col in display_df.columns:
            if display_df[col].dtype == 'float64':
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
                
        print("\n" + "=" * 110)
        print("       Software Multi-Class Rash Driving Classification — TOPSIS Performance Metrics")
        print("       (6-class: Normal, Lane Weaving, Lane Swerving, Hard Braking, Hard Cornering, Quick U-turn)")
        print("=" * 110)
        print(display_df.to_string(index=False))
        print("=" * 110)
        
        if 'Rank' in topsis_df.columns:
            winner_row = topsis_df[topsis_df['Rank'] == 1].iloc[0]
            best_model_name = winner_row['Model']
            print(f"\n🏆 ULTIMATE WINNER (TOPSIS Rank #1): {best_model_name}")
            print(f"   Topsis Score: {winner_row.get('Topsis Score', 'N/A'):.4f}")
        else:
            best_model_name = topsis_df.iloc[0, 0] 
            print(f"\n🏆 ULTIMATE WINNER: {best_model_name}")
            
    except Exception as e:
        print(f"\n⚠️ WARNING: Auto-TOPSIS failed ({e}). Falling back to raw F1-Score.")
        results_df = results_df.sort_values("Macro F1-Score", ascending=False).reset_index(drop=True)
        print(results_df.to_string(index=False))
        best_model_name = results_df.iloc[0]['Model']
        print(f"\n🏆 BEST MODEL (Raw F1): {best_model_name}")
    
    # ── EXPORT TRUE BEST MODEL ──
    out_dir = Path(cfg["paths"]["model_checkpoints"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    best_model = models[best_model_name]
    
    model_path = out_dir / "best_rash_classifier.pkl"
    joblib.dump(best_model, model_path)
    joblib.dump(le, out_dir / "rash_label_encoder.pkl")
    
    print(f"\n=> 💾 Successfully exported the TOPSIS-verified best model ({best_model_name}) and label encoder to:")
    print(f"   {model_path}")

if __name__ == "__main__":
    main()
