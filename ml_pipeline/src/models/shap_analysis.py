"""
IntelliCrash — Explainable AI (SHAP) Analysis Script

Generates SHAP Global Feature Importance bar plots and 
local Waterfall plots for the XGBoost Rash Driving Classifier.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import xgboost as xgb
import shap
import joblib
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config
from src.features.feature_engineering import FEATURE_NAMES

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 300, "font.family": "serif"})

def ensure_plot_dir():
    cfg = get_config()
    out = Path(cfg["paths"]["plots_dir"]) / "shap"
    out.mkdir(parents=True, exist_ok=True)
    return out

def run_shap_analysis():
    out_dir = ensure_plot_dir()
    cfg = get_config()
    
    print("Loading engineered features and metadata for SHAP background data...")
    try:
        processed_dir = Path(cfg["paths"]["processed_dir"])
        windows_dir = Path(cfg["paths"]["windows_dir"])
        
        X_xgb = np.load(processed_dir / "imu_features.npy")
        meta_df = pd.read_parquet(windows_dir / "imu_metadata.parquet")
        y_xgb = meta_df['rash_class'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb)
    except FileNotFoundError:
        print("Error: Required dataset files not found. Please run preprocessing first.")
        return
        
    print("Loading Best Auto-TOPSIS Model...")
    model_dir = Path(cfg["paths"]["model_checkpoints"])
    clf = joblib.load(model_dir / "best_rash_classifier.pkl")
    le = joblib.load(model_dir / "rash_label_encoder.pkl")
    
    # 1. Initialize SHAP Explainer (auto-detect best method for the model)
    print("Calculating SHAP values...")
    import warnings
    warnings.filterwarnings("ignore")
    
    # Try TreeExplainer first (fast, works for LightGBM/XGBoost/CatBoost/RF/etc.)
    # Fall back to KernelExplainer for non-tree models (MLP, SVM, etc.)
    try:
        # For pipeline models, extract the underlying estimator
        model_for_shap = clf
        if hasattr(clf, 'named_steps'):
            # sklearn Pipeline — get the last step
            model_for_shap = clf[-1]
        elif hasattr(clf, 'final_estimator_'):
            # StackingClassifier — use the final estimator
            model_for_shap = clf.final_estimator_
        
        explainer = shap.TreeExplainer(model_for_shap)
        print("  Using TreeExplainer (fast, exact).")
        use_tree = True
    except Exception:
        print("  TreeExplainer not supported for this model. Using KernelExplainer (slower)...")
        background = shap.kmeans(X_train, 10)
        explainer = shap.KernelExplainer(lambda x: clf.predict_proba(x), background)
        use_tree = False
    
    # Take a subsample for global plotting to prevent freezing
    X_test_sub = X_test[:50] 
    y_test_sub = y_test[:50]
    
    CLASS_NAMES = {
        0: "Normal Driving",
        1: "Lane Weaving",
        2: "Lane Swerving",
        3: "Hard Braking",
        4: "Hard Cornering",
        5: "Quick U-turn"
    }
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_test_sub)
    
    # 2. Global Feature Importance (Bar Plot)
    plt.figure()
    shap.summary_plot(shap_values, X_test_sub, feature_names=FEATURE_NAMES, class_names=le.classes_, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance: Best Auto-TOPSIS Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_global_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Waterfall Plot for a specific RASH MANEUVER prediction
    # Find a rash driving sample from the test set to guarantee we explain an interesting event
    rash_indices = np.where(y_test != 0)[0]
    if len(rash_indices) > 0:
        idx = rash_indices[0]
        maneuver_name = CLASS_NAMES.get(y_test[idx], "Rash Maneuver")
        title = f"XAI (Best Auto-TOPSIS Model): {maneuver_name} Incident Reasoning"
    else:
        idx = 0
        maneuver_name = CLASS_NAMES.get(y_test[idx], "Normal Driving")
        title = f"XAI (Best Auto-TOPSIS Model): Maneuver Reasoning ({maneuver_name})"
        
    crash_sample = X_test[idx:idx+1]
    
    # Calculate SHAP values explicitly for this one guaranteed sample
    shap_val_single = explainer.shap_values(crash_sample)
    
    # Handle list vs 3D array output from different SHAP versions/explainers
    target_class = y_test[idx]
    if isinstance(shap_val_single, list):
        vals = shap_val_single[target_class][0]
    elif isinstance(shap_val_single, np.ndarray) and shap_val_single.ndim == 3:
        # shape: (num_samples, num_features, num_classes)
        vals = shap_val_single[0, :, target_class]
    else:
        # fallback for binary or flat array
        vals = shap_val_single[0]
        
    # Handle expected_value format differences
    if isinstance(explainer.expected_value, list):
        base_val = explainer.expected_value[target_class]
    elif isinstance(explainer.expected_value, np.ndarray) and explainer.expected_value.size > 1:
        base_val = explainer.expected_value[target_class]
    else:
        base_val = explainer.expected_value

    exp = shap.Explanation(values=vals, 
                           base_values=base_val, 
                           data=crash_sample[0], 
                           feature_names=FEATURE_NAMES)
        
    plt.figure(figsize=(10, 6))
    try:
        shap.plots.waterfall(exp, show=False)
    except AttributeError:
        shap.waterfall_plot(exp, show=False)
    plt.title(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_waterfall_example.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # --- NEW: Generate Human-Readable PDF Text Report ---
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    shap_vals = exp.values
    feature_names = exp.feature_names
    data_vals = exp.data
    
    top_indices = np.argsort(-np.abs(shap_vals))
    
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_xy(10, 15)
        pdf.cell(190, 10, "IntelliCrash Explainable AI (XAI) Forensic Report", align="C")
        
        # Subtitle
        pdf.set_font("Helvetica", "B", 14)
        pdf.set_xy(10, 30)
        
        class_name_upper = CLASS_NAMES.get(y_test[idx], "UNKNOWN").upper()
        
        if idx == 6:
            pdf.set_text_color(200, 0, 0) # Red for CRASH
            pdf.cell(190, 10, "Incident Severity: CRASH DETECTED", align="L")
        else:
            pdf.set_text_color(0, 0, 200) # Blue for Maneuver
            pdf.cell(190, 10, f"Incident Severity: MANEUVER ({class_name_upper}) DETECTED", align="L")
            
        pdf.set_text_color(0, 0, 0)
        
        # Body Text
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_xy(10, 45)
        pdf.cell(190, 10, "AI Decision Reasoning:")
        
        pdf.set_font("Helvetica", "", 12)
        pdf.set_xy(10, 55)
        pdf.multi_cell(190, 8, txt="The XGBoost model classified this event based on the following primary physical factors:")
        
        # Manually track Y to prevent any fpdf2 text-wrapping overlap bugs
        current_y = pdf.get_y() + 2
        
        for i in range(3):
            top_idx = top_indices[i]
            feat = feature_names[top_idx]
            val = data_vals[top_idx]
            impact = shap_vals[top_idx]
            direction = "increased" if impact > 0 else "decreased"
            line = f"- {feat} was measured at {val:.2f}, which {direction} the raw hazard score by {abs(impact):.2f} points."
            
            pdf.set_xy(10, current_y)
            pdf.multi_cell(190, 8, txt=line)
            current_y = pdf.get_y() + 2
            
        # Explaining the Waterfall Plot
        current_y += 5
        pdf.set_xy(10, current_y)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(190, 8, "Understanding the Visual Chart Below:")
        
        current_y += 8
        pdf.set_xy(10, current_y)
        pdf.set_font("Helvetica", "I", 11)
        explanation = "The chart visualizes the mathematical breakdown of the AI's decision. The gray E[f(X)] at the bottom is the baseline average score. The red bars push the hazard score higher (more dangerous), while the blue bars push the score lower (safer). The final f(x) at the top is the absolute hazard rating assigned to this event."
        pdf.multi_cell(190, 6, txt=explanation)
        
        # Embed SHAP Waterfall Plot
        current_y = pdf.get_y() + 10
        img_path = str(out_dir / "shap_waterfall_example.png")
        if os.path.exists(img_path):
            # Add image, scaled to fit A4 width
            pdf.image(img_path, x=10, y=current_y, w=190)
            
        # Move Y to below the image (assuming roughly 110mm height for the image)
        current_y += 120 
        if current_y > 270: # If we bleed off page, add new page
            pdf.add_page()
            current_y = 15
            
        # Conclusion Footer
        pdf.set_xy(10, current_y)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(190, 6, txt="Conclusion: This forensic report is generated autonomously by the SHAP TreeExplainer to provide transparency for emergency dispatchers and insurance evaluators.")
        
        pdf_path = reports_dir / "crash_forensic_report.pdf"
        pdf.output(str(pdf_path))
        print(f"Human-readable forensic PDF report saved to {pdf_path}")
        
    except ImportError:
        print("fpdf2 library not found. Please run: pip install fpdf2")
    
    print(f"SHAP Analysis complete. Plots saved to {out_dir}")

if __name__ == "__main__":
    run_shap_analysis()
