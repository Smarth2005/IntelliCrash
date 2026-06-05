"""
Export trained IntelliCrash Bi-LSTM model to ONNX.
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.config import get_config
from src.models.lstm import IntelliCrashLSTM


def export_model():
    cfg = get_config()
    
    # Init model
    model = IntelliCrashLSTM(
        input_size=24,
        hidden_size=cfg["lstm"]["hidden_size"],
        num_layers=cfg["lstm"]["num_layers"],
        dropout=cfg["lstm"]["dropout"]
    )
    
    # Load weights
    checkpoint_path = Path(cfg["paths"]["model_checkpoints"]) / "best_lstm.pth"
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.eval()
    
    print(f"Loaded weights from {checkpoint_path}")
    
    out_dir = Path(cfg["paths"]["model_onnx"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Dummy input
    dummy_input = torch.randn(1, 200, 24)
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
    
    print(f"ONNX model exported to: {onnx_path}")


if __name__ == "__main__":
    export_model()
