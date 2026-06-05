"""
IntelliCrash — Bi-LSTM Crash Detection Model.

A PyTorch implementation of the dual-head Bi-LSTM network:
- Input: Windowed sensor data (batch, seq_len, features)
- Output 1: Binary crash probability (0-1)
- Output 2: Severity score regression (0-1)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class IMUDataset(Dataset):
    """PyTorch Dataset for windowed IMU data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, severity: np.ndarray = None):
        """
        Args:
            X: Array of shape (num_samples, seq_len, features)
            y: Binary crash labels (num_samples,)
            severity: Optional severity scores (num_samples,). Defaults to y.
        """
        self.X = torch.FloatTensor(X)
        self.y_crash = torch.FloatTensor(y).unsqueeze(1)
        
        if severity is not None:
            self.y_severity = torch.FloatTensor(severity).unsqueeze(1)
        else:
            self.y_severity = self.y_crash.clone()
            
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return {
            'features': self.X[idx],
            'crash': self.y_crash[idx],
            'severity': self.y_severity[idx]
        }


class FocalLoss(nn.Module):
    """Focal Loss for handling severe class imbalance."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class IntelliCrashLSTM(nn.Module):
    """
    Bi-LSTM Network with dual output heads:
    1. Crash classification (Sigmoid)
    2. Severity regression (Sigmoid/Linear)
    """
    def __init__(
        self, 
        input_size: int = 24, # 6 raw + 18 engineered (or whatever feature count we use)
        hidden_size: int = 128, 
        num_layers: int = 2, 
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 1D Conv layer for initial feature extraction/smoothing
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=input_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Bi-LSTM Backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Shared FC layer
        self.fc_shared = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Head 1: Crash Classification
        self.head_crash = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Head 2: Severity Regression
        self.head_severity = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()  # output [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, features)
        Returns:
            crash_prob: (batch, 1)
            severity_score: (batch, 1)
        """
        # PyTorch Conv1d expects (batch, channels, seq_len)
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)
        x = x_conv.transpose(1, 2)
        
        # LSTM output: (batch, seq_len, hidden_size * 2)
        lstm_out, _ = self.lstm(x)
        
        # Attention pooling
        attn_weights = self.attention(lstm_out)
        # Context vector: weighted sum over seq_len
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Shared features
        shared = self.fc_shared(context)
        
        # Dual heads
        crash_prob = self.head_crash(shared)
        severity_score = self.head_severity(shared)
        
        return crash_prob, severity_score

    def get_attention_weights(self, x):
        """Helper to extract attention weights for explainability."""
        x_conv = x.transpose(1, 2)
        x_conv = self.conv1d(x_conv)
        x = x_conv.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        return self.attention(lstm_out)


if __name__ == "__main__":
    # Quick test
    batch_size = 16
    seq_len = 200
    features = 24
    
    model = IntelliCrashLSTM(input_size=features)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    dummy_x = torch.randn(batch_size, seq_len, features)
    crash, sev = model(dummy_x)
    print(f"Crash out: {crash.shape}")
    print(f"Severity out: {sev.shape}")
