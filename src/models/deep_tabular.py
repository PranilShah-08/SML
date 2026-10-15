"""
Deep Tabular Neural Network Baselines for Network Intrusion Detection.
Implements:
1. MLP (Multi-Layer Perceptron with BatchNorm, Dropout)
2. 1D-CNN (Temporal/Feature ConvNet)
3. LSTM (Recurrent Tabular Model)
4. FT-Transformer (Feature Tokenizer + Multi-Head Self Attention)
5. TabNet (via pytorch_tabnet)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional
from pytorch_tabnet.tab_model import TabNetClassifier

class MLPTabular(nn.Module):
    def __init__(self, in_features: int, out_classes: int = 2, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_classes)
        )

    def forward(self, x):
        return self.net(x)

class CNN1DTabular(nn.Module):
    def __init__(self, in_features: int, out_classes: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, out_classes)
        )

    def forward(self, x):
        # x: (B, D) -> (B, 1, D)
        x = x.unsqueeze(1)
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)

class LSTMTabular(nn.Module):
    def __init__(self, in_features: int, out_classes: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.n_tokens = 8
        self.token_dim = 16
        self.proj = nn.Linear(in_features, self.n_tokens * self.token_dim)
        self.lstm = nn.LSTM(input_size=self.token_dim, hidden_size=hidden_dim, batch_first=True, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_classes)
        )

    def forward(self, x):
        B = x.size(0)
        tokens = self.proj(x).view(B, self.n_tokens, self.token_dim)
        _, (h_n, _) = self.lstm(tokens)
        return self.fc(h_n[-1])

class FTTransformerLite(nn.Module):
    """
    Feature Tokenizer Transformer for Tabular Data.
    Projects each continuous feature into an embedding vector and applies self-attention.
    """
    def __init__(self, in_features: int, out_classes: int = 2, d_token: int = 32, n_heads: int = 4):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(in_features, d_token))
        self.bias = nn.Parameter(torch.Tensor(in_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token*2, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Linear(d_token, out_classes)
        )

    def forward(self, x):
        # x: (B, D) -> token: (B, D, d_token)
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        out = self.transformer(tokens)
        cls_out = out[:, 0, :]
        return self.head(cls_out)

class PyTorchTabularClassifier:
    """
    Unified Scikit-Learn wrapper for PyTorch tabular architectures.
    """
    def __init__(self, model_type: str = 'mlp', in_features: int = 25, out_classes: int = 2,
                 epochs: int = 10, batch_size: int = 128, lr: float = 1e-3, seed: int = 42):
        self.model_type = model_type
        self.in_features = in_features
        self.out_classes = out_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        torch.manual_seed(seed)

        if model_type == 'mlp':
            self.model = MLPTabular(in_features, out_classes)
        elif model_type == 'cnn':
            self.model = CNN1DTabular(in_features, out_classes)
        elif model_type == 'lstm':
            self.model = LSTMTabular(in_features, out_classes)
        elif model_type == 'ft_transformer':
            self.model = FTTransformerLite(in_features, out_classes)
        else:
            raise ValueError(f"Unknown deep tabular model type: {model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        # Class weights for focal / balanced loss
        unique, counts = np.unique(y, return_counts=True)
        weights = len(y) / (len(unique) * counts.astype(float))
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        self.model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                logits = self.model(bx)
                loss = criterion(logits, by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.model(x_tensor)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
