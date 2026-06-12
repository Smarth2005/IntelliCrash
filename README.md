## Repository Structure 
```Plaintext
IntelliCrash/
├── configs/                                 # Configuration files (hyperparameters, paths)
│   └── config.yaml                          # Global project settings
│
├── dashboard/                               # Next.js web application for real-time visualization
│
├── data/                                    # Data pipeline outputs
│   └── processed/                           # Extracted features and windowed arrays
│       ├── imu_csi_scores.npy               # Physics-based crash severity scores
│       ├── imu_features.npy                 # Engineered temporal/frequency features
│       └── windows/                         # 2-second overlapping sliding windows
│           ├── imu_augmented_metadata.parquet
│           ├── imu_augmented_X.npy          # SMOTE augmented feature tensors (Ignored in Git)
│           ├── imu_augmented_X_test.npy     # Test split for evaluation (Ignored in Git)
│           ├── imu_augmented_y.npy          # Training labels
│           ├── imu_augmented_y_test.npy     # Test labels
│           ├── imu_metadata.parquet         # Base metadata
│           ├── imu_X.npy                    # Original un-augmented tensors (Ignored in Git)
│           └── imu_y.npy                    # Original labels
│
├── dataset/                                 # Raw IMU dataset (Ignored in Git due to size)
│
├── models/                                  # Trained model artifacts
│   ├── checkpoints/                         # PyTorch & XGBoost model weights
│   │   ├── best_lstm.pth                    # Optimal Phase-2 Bi-LSTM weights
│   │   ├── rash_label_encoder.pkl           # Encoded labels for Phase-1
│   │   └── xgboost_rash_classifier.json     # Trained Phase-1 Rash Driving classifier
│   └── onnx/                                # Exported models for edge deployment
│       ├── intellicrash_lstm.onnx           # ONNX computational graph
│       └── intellicrash_lstm.onnx.data      # ONNX external tensor weights
│
├── notebooks/                               # Jupyter/Colab notebooks
│   └── Model_Training_Pipeline.ipynb        # Cloud training End-to-End Pipeline
│
├── outputs/                                 # Generated evaluation metrics and visuals
│   ├── plots/                               # Confusion matrices, ROC curves, etc.
│   ├── reports/                             # Classification reports and logs
│   ├── shap/                                # Model interpretability (SHAP values)
│   ├── system_architecture_final.png        # Architecture diagram
│   └── system_flow_diagram.png              # System flow diagram
│
├── src/                                     # Main source code
│   ├── data/                                # Preprocessing and data manipulation
│   │   ├── eda.py                           # Exploratory Data Analysis script
│   │   ├── export_dataset_csv.py            # Generates the panel-ready CSV dataset
│   │   ├── preprocess_imu.py                # IMU cleaning and windowing logic
│   │   └── synthetic_crashes.py             # SMOTE implementation for class balancing
│   ├── edge/                                # Edge device (Raspberry Pi/C++) deployment
│   │   ├── dispatch_alert.py                # Cloud/Emergency API dispatch logic
│   │   └── inference.cpp                    # C++ ONNX runtime inference script
│   ├── features/                            # Feature engineering logic
│   │   └── feature_engineering.py           # Physics CSI calculation (Equation 4)
│   ├── models/                              # ML model definitions and training logic
│   │   ├── evaluate_ablation.py             # Dual-gate architecture evaluation script
│   │   ├── export_onnx.py                   # Converts PyTorch model to ONNX format
│   │   ├── lstm.py                          # Bi-LSTM Neural Network PyTorch definition
│   │   ├── train_lstm.py                    # Phase 2: LSTM training loop
│   │   └── train_rash_classifier.py         # Phase 1: XGBoost classifier training
│   └── utils/                               # Helper functions
│       └── config.py                        # YAML configuration loader
│
├── IntelliCrash_Dataset.csv                 # Preprocessed, panel-ready feature dataset
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```
