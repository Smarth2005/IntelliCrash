"""
IntelliCrash — Phase-1 Pre-Crash Monitoring: Rash Driving Classifier (Sensor Evaluation)
==========================================================================================
Trains and evaluates 12 baseline models on the real-world sensor dataset:
  - Normal Driving (from verified sensor sessions: 0006, 0008, 0009, 0011, 0012)
  - Rash Driving Maneuvers (from pre_crash/imu_session_0015.csv)

Follows the identical data pipeline:
  1. 5th-order zero-phase Butterworth low-pass filtering (1.3 Hz cutoff)
  2. 2-second (200-sample) sliding windows (50% overlap / 100-sample stride)
  3. 26 physics-engineered features
  4. 12 baseline models + RandomizedSearchCV hyperparameter tuning
  5. Multi-criteria mathematical ranking via TOPSIS
  6. SHAP Explainable AI (XAI) feature attribution & Forensic PDF Report generation
"""

import os
import sys
import glob
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import make_pipeline

# Model Libraries
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
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform
from scipy.signal import butter, filtfilt

try:
    from feature_engineering import extract_features_single, FEATURE_NAMES
except ImportError:
    from hardware.feature_engineering import extract_features_single, FEATURE_NAMES

def set_all_seeds(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)

set_all_seeds(42)

BUTTER_ORDER = 5
BUTTER_CUTOFF = 1.3
SAMPLING_RATE = 100.0
WINDOW_SIZE = 200
STRIDE = 100


def butterworth_lowpass(data: np.ndarray, cutoff: float = BUTTER_CUTOFF, fs: float = SAMPLING_RATE, order: int = BUTTER_ORDER) -> np.ndarray:
    if len(data) <= order * 3:
        return data.copy()
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    mean_val = np.mean(data)
    centered = data - mean_val
    filtered = filtfilt(b, a, centered)
    return filtered + mean_val


def extract_windows_and_features(df_stream: pd.DataFrame, label_str: str):
    """Segment a continuous stream into filtered 6-channel windows and compute 26 physics features."""
    ax = df_stream["accel_x_m_s2"].values.astype(np.float32)
    ay = df_stream["accel_y_m_s2"].values.astype(np.float32)
    gz = df_stream["gyro_z_rad_s"].values.astype(np.float32)

    filt_ax = butterworth_lowpass(ax)
    filt_ay = butterworth_lowpass(ay)
    filt_gz = butterworth_lowpass(gz)

    n_samples = len(ax)
    feats_list = []
    labels_list = []

    for start_idx in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
        end_idx = start_idx + WINDOW_SIZE
        w_ax = ax[start_idx:end_idx]
        w_ay = ay[start_idx:end_idx]
        w_gz = gz[start_idx:end_idx]
        w_fax = filt_ax[start_idx:end_idx]
        w_fay = filt_ay[start_idx:end_idx]
        w_fgz = filt_gz[start_idx:end_idx]

        raw_window = np.column_stack([w_ax, w_ay, w_gz, w_fax, w_fay, w_fgz]).astype(np.float32)
        features = extract_features_single(raw_window, fs=SAMPLING_RATE)
        feats_list.append(features)
        labels_list.append(label_str)

    return feats_list, labels_list


def build_sensor_dataset():
    """Load pre_crash/imu_session_0015.csv and normal sensor sessions to construct feature matrix."""
    print("\n--- [Step 1/5] Building Real Sensor Dataset (Phase-1 Pre-Crash) ---")
    
    # 1. Load Rash Driving Maneuvers from pre_crash/imu_session_0015.csv
    file_rash = Path("pre_crash/imu_session_0015.csv")
    if not file_rash.exists():
        raise FileNotFoundError(f"Cannot find {file_rash}")

    print(f"Loading Rash Driving Telemetry: {file_rash}...")
    df_rash = pd.read_csv(file_rash)
    # Filter to rash rows only (skip first stationary 141 rows)
    df_rash_valid = df_rash[df_rash["label"] == "rash_drive"].reset_index(drop=True)
    print(f"  Extracted {len(df_rash_valid):,} samples of active rash driving maneuvers.")

    rash_feats, rash_labels = extract_windows_and_features(df_rash_valid, "Rash Driving Maneuver")
    print(f"  => Created {len(rash_feats):,} sliding windows for Rash Driving Maneuver.")

    # 2. Load Normal Driving from latest sensor data
    normal_files = sorted(glob.glob("latest sensor data/*.csv"))
    print(f"\nLoading Normal Driving Telemetry from {len(normal_files)} sensor sessions...")
    normal_feats = []
    normal_labels = []

    for nf in normal_files:
        df_norm = pd.read_csv(nf)
        df_norm_clean = df_norm[df_norm["label"] == "non_crash"].reset_index(drop=True)
        if len(df_norm_clean) >= WINDOW_SIZE:
            n_feats, n_labels = extract_windows_and_features(df_norm_clean, "Normal Driving")
            normal_feats.extend(n_feats)
            normal_labels.extend(n_labels)

    print(f"  => Created {len(normal_feats):,} sliding windows for Normal Driving.")

    # Subsample normal windows to maintain balanced ratio with rash windows
    if len(normal_feats) > len(rash_feats) * 1.5:
        np.random.seed(42)
        target_norm = int(len(rash_feats) * 1.25)
        keep_indices = np.random.choice(len(normal_feats), size=target_norm, replace=False)
        normal_feats = [normal_feats[i] for i in sorted(keep_indices)]
        normal_labels = [normal_labels[i] for i in sorted(keep_indices)]
        print(f"  Balanced Normal Driving windows: {len(normal_feats):,}")

    X = np.array(normal_feats + rash_feats, dtype=np.float32)
    y = np.array(normal_labels + rash_labels)

    print(f"\nFinal Real Sensor Dataset Matrix:")
    print(f"  Features shape: {X.shape} (26 physical telemetry features)")
    print(f"  Class Distribution: {pd.Series(y).value_counts().to_dict()}")

    return X, y


def run_baselines_and_topsis(X, y):
    """Train 12 ML baselines, evaluate, and rank with TOPSIS."""
    print("\n--- [Step 2/5] Training 12 Baseline Models with RandomizedSearchCV ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    num_classes = len(le.classes_)

    print(f"Training split: {len(X_train):,} windows | Testing split: {len(X_test):,} windows")
    print(f"Classes encoded: {dict(enumerate(le.classes_))}")

    sample_weights = compute_sample_weight("balanced", y_train_enc)

    # 1. XGBoost Hyperparameter Tuning
    print("\n[Tuning] Optimizing XGBoost with RandomizedSearchCV (3-fold CV)...")
    xgb_param_dist = {
        "n_estimators": randint(150, 600),
        "max_depth": randint(3, 8),
        "learning_rate": uniform(0.02, 0.15),
        "subsample": uniform(0.7, 0.3),
        "colsample_bytree": uniform(0.7, 0.3),
        "min_child_weight": randint(1, 6),
    }

    xgb_base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_param_dist, n_iter=25, cv=3,
        scoring="f1", random_state=42, n_jobs=-1, verbose=0
    )
    xgb_search.fit(X_train, y_train_enc, sample_weight=sample_weights)
    best_xgb_params = xgb_search.best_params_
    print(f"  Best XGBoost Params: {best_xgb_params}")
    print(f"  Best CV F1: {xgb_search.best_score_:.4f}")

    # Define all 12 models
    models = {
        "XGBoost (Default)": xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=150,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
        "XGBoost (Tuned)": xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
            tree_method="hist",
            **best_xgb_params,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_depth=6, random_state=42
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=45,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=400,
            learning_rate=0.05,
            depth=7,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
            thread_count=-1,
        ),
        "MLP (Neural Network)": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=400,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.15,
                learning_rate="adaptive",
            ),
        ),
        "SVC (SVM)": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10.0, class_weight="balanced", random_state=42, probability=True),
        ),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1),
        ),
    }

    # Build Ensembles
    estimators = [
        ("knn", models["K-Nearest Neighbors"]),
        ("rf", models["Random Forest"]),
        ("lgbm", models["LightGBM"]),
        ("xgb", models["XGBoost (Tuned)"]),
    ]

    models["Stacking Ensemble"] = StackingClassifier(
        estimators=estimators,
        final_estimator=xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1),
        n_jobs=-1,
    )

    models["Soft Voting Ensemble"] = VotingClassifier(
        estimators=[
            ("lgbm", models["LightGBM"]),
            ("xgb", models["XGBoost (Tuned)"]),
            ("rf", models["Random Forest"]),
        ],
        voting="soft",
        n_jobs=-1,
    )

    print("\n--- [Step 3/5] Evaluating All 12 Baselines ---")
    results = []

    for name, model in tqdm(models.items(), desc="Training & Evaluating"):
        if "XGBoost" in name or "Random Forest" in name or "Extra Trees" in name or "CatBoost" in name or "LightGBM" in name:
            try:
                model.fit(X_train, y_train_enc, sample_weight=sample_weights)
            except Exception:
                model.fit(X_train, y_train_enc)
        else:
            model.fit(X_train, y_train_enc)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test_enc, y_pred)
        prec = precision_score(y_test_enc, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test_enc, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test_enc, y_pred, average="macro", zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Macro Precision": prec,
            "Macro Recall": rec,
            "Macro F1-Score": f1,
        })

    results_df = pd.DataFrame(results)
    
    cfg = get_config()
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    input_csv = reports_dir / "sensor_rash_topsis_input.csv"
    results_df.to_csv(input_csv, index=False)

    print("\n--- [Step 4/5] Multi-Criteria TOPSIS Mathematical Ranking ---")
    topsis_result_csv = reports_dir / "sensor_rash_topsis_result.csv"
    topsis_script = str(Path(__file__).resolve().parent.parent.parent / "topsis.py")

    weights = "0.30,0.15,0.15,0.40"
    impacts = "+,+,+,+"

    subprocess.run([
        sys.executable, topsis_script,
        str(input_csv), weights, impacts, str(topsis_result_csv)
    ], check=True)

    topsis_df = pd.read_csv(topsis_result_csv)
    
    # Formatted display
    display_df = topsis_df.copy()
    for col in display_df.columns:
        if display_df[col].dtype == "float64":
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

    print("\n" + "=" * 110)
    print("      Real-World Sensor Rash Driving Classification — TOPSIS Performance Metrics")
    print("      (Evaluated on RC Buggy ISM330DHCX IMU Telemetry: Normal vs. Rash Driving Maneuver)")
    print("=" * 110)
    print(display_df.to_string(index=False))
    print("=" * 110)

    winner_row = topsis_df[topsis_df["Rank"] == 1].iloc[0]
    best_model_name = winner_row["Model"]
    print(f"\n[WINNER] ULTIMATE WINNER (TOPSIS Rank #1): {best_model_name}")
    print(f"   Accuracy: {winner_row['Accuracy']:.4f} | Macro F1: {winner_row['Macro F1-Score']:.4f} | TOPSIS Score: {winner_row['Topsis Score']:.4f}")

    # Export best model
    out_dir = Path(cfg["paths"]["model_checkpoints"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_model = models[best_model_name]
    model_path = out_dir / "best_sensor_rash_classifier.pkl"
    joblib.dump(best_model, model_path)
    joblib.dump(le, out_dir / "sensor_rash_label_encoder.pkl")
    print(f"\n[SAVED] Saved best sensor model to: {model_path}")

    return best_model, le, X_train, X_test, y_train_enc, y_test_enc


def run_sensor_shap_and_report(best_model, le, X_train, X_test, y_train_enc, y_test_enc):
    """Run SHAP Explainable AI and generate human-readable XAI Forensic Report."""
    print("\n--- [Step 5/5] Explainable AI (SHAP) Analysis & Forensic PDF Generation ---")
    import shap
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"figure.dpi": 300, "font.family": "serif"})

    cfg = get_config()
    shap_plot_dir = Path(cfg["paths"]["plots_dir"]) / "shap"
    shap_plot_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(cfg["paths"]["reports_dir"])

    # Extract base estimator if pipeline or stacking
    model_for_shap = best_model
    if hasattr(best_model, "named_steps"):
        model_for_shap = best_model[-1]
    elif hasattr(best_model, "final_estimator_"):
        model_for_shap = best_model.final_estimator_

    print("Computing SHAP values using TreeExplainer...")
    try:
        explainer = shap.TreeExplainer(model_for_shap)
    except Exception:
        background = shap.kmeans(X_train, 10)
        explainer = shap.KernelExplainer(lambda x: best_model.predict_proba(x), background)

    # Global Feature Importance
    X_sub = X_test[:60]
    shap_values = explainer.shap_values(X_sub)

    plt.figure(figsize=(10, 8))
    # If list, take class 1 (Rash Driving Maneuver)
    shap_plot_vals = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values
    shap.summary_plot(
        shap_plot_vals, X_sub, feature_names=FEATURE_NAMES, plot_type="bar", show=False
    )
    plt.title("SHAP Global Physical Feature Importance (Sensor Rash Classifier)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    global_img_path = shap_plot_dir / "sensor_shap_global_feature_importance.png"
    plt.savefig(global_img_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {global_img_path}")

    # Local Waterfall Plot for a verified Rash Driving Maneuver from Session 0015
    rash_indices = np.where(y_test_enc == 1)[0]
    idx = rash_indices[0] if len(rash_indices) > 0 else 0
    sample_window = X_test[idx:idx + 1]

    shap_val_single = explainer.shap_values(sample_window)

    if isinstance(shap_val_single, list):
        vals = shap_val_single[1][0] if len(shap_val_single) > 1 else shap_val_single[0][0]
    elif isinstance(shap_val_single, np.ndarray) and shap_val_single.ndim == 3:
        vals = shap_val_single[0, :, 1]
    elif isinstance(shap_val_single, np.ndarray) and shap_val_single.ndim == 2:
        vals = shap_val_single[0]
    else:
        vals = shap_val_single

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1] if len(base_val) > 1 else base_val[0])
    else:
        base_val = float(base_val)

    exp = shap.Explanation(
        values=vals,
        base_values=base_val,
        data=sample_window[0],
        feature_names=FEATURE_NAMES,
    )

    plt.figure(figsize=(11, 7))
    try:
        shap.plots.waterfall(exp, show=False)
    except AttributeError:
        shap.waterfall_plot(exp, show=False)

    plt.title("XAI Physical Telemetry Reasoning: Rash Driving Maneuver Detected", fontsize=14, fontweight="bold")
    plt.tight_layout()
    waterfall_img_path = shap_plot_dir / "sensor_shap_waterfall_example.png"
    plt.savefig(waterfall_img_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {waterfall_img_path}")

    # Generate Human-Readable PDF Forensic Report
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()

        # Header
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_xy(10, 15)
        pdf.cell(190, 10, "IntelliCrash Explainable AI (XAI) Forensic Report", align="C")

        pdf.set_font("Helvetica", "B", 13)
        pdf.set_xy(10, 28)
        pdf.set_text_color(220, 38, 38) # Red alert
        pdf.cell(190, 10, "INCIDENT SEVERITY: RASH DRIVING MANEUVER DETECTED", align="L")
        pdf.set_text_color(0, 0, 0)

        pdf.set_font("Helvetica", "", 11)
        pdf.set_xy(10, 40)
        pdf.multi_cell(
            190, 6,
            txt="An aggressive driving maneuver was captured by the onboard ISM330DHCX IMU on the RC testbed. "
                "The buzzer sounded to alert the driver. Below is the autonomous physical feature attribution breakdown:"
        )

        # Top 3 physical drivers
        top_indices = np.argsort(-np.abs(exp.values))[:3]
        cur_y = pdf.get_y() + 4

        pdf.set_font("Helvetica", "B", 11)
        pdf.set_xy(10, cur_y)
        pdf.cell(190, 8, "Primary Physical Driving Factors Responsible:")
        cur_y += 8

        pdf.set_font("Helvetica", "", 10)
        for rank, ti in enumerate(top_indices, start=1):
            feat = exp.feature_names[ti]
            val = exp.data[ti]
            impact = exp.values[ti]
            direction = "pushed hazard score higher (+)" if impact > 0 else "reduced hazard score (-)"
            txt_line = f"{rank}. {feat} = {val:.2f} -> {direction} by {abs(impact):.2f} pts"
            pdf.set_xy(12, cur_y)
            pdf.cell(186, 6, txt_line)
            cur_y += 6

        # Embed Waterfall
        cur_y += 6
        if waterfall_img_path.exists():
            pdf.image(str(waterfall_img_path), x=10, y=cur_y, w=190)
            cur_y += 125

        if cur_y > 270:
            pdf.add_page()
            cur_y = 15

        pdf.set_xy(10, cur_y)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(190, 5, "Conclusion: This autonomous XAI forensic report explains the physical kinematics that tripped the edge safety alert, providing auditable verification for driver training and telematics.")

        pdf_report_path = reports_dir / "sensor_rash_forensic_report.pdf"
        pdf.output(str(pdf_report_path))
        print(f"  Saved Forensic Report: {pdf_report_path}")

    except Exception as e:
        print(f"  Notice on PDF generation: {e}")

    print("\n[SUCCESS] Full Phase-1 Sensor Evaluation & XAI Complete!")


def main():
    X, y = build_sensor_dataset()
    best_model, le, X_train, X_test, y_train_enc, y_test_enc = run_baselines_and_topsis(X, y)
    run_sensor_shap_and_report(best_model, le, X_train, X_test, y_train_enc, y_test_enc)


if __name__ == "__main__":
    main()

