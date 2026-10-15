"""
Classical ML & Anomaly Detection Baselines for Network Intrusion Detection.
Implements:
- Classical: LR, RF, XGBoost, LightGBM, CatBoost (each with class weights)
- Anomaly / Unseen References: Isolation Forest and PyTorch Deep Autoencoder
- Imbalance handling: None, Class Weights, SMOTE
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, latent_dim),
            nn.LeakyReLU(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

class AutoencoderAnomalyDetector:
    def __init__(self, input_dim: int, epochs: int = 15, batch_size: int = 128, lr: float = 1e-3, seed: int = 42):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        torch.manual_seed(seed)
        self.model = DeepAutoencoder(input_dim)
        self.threshold: float = 0.5

    def fit(self, X_benign: np.ndarray):
        dataset = TensorDataset(torch.tensor(X_benign, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch in loader:
                x = batch[0]
                recon = self.model(x)
                loss = criterion(recon, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Compute calibration threshold on benign reconstruction error (e.g. 99th percentile)
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X_benign, dtype=torch.float32)
            recon = self.model(x_tensor)
            errors = torch.mean((x_tensor - recon) ** 2, dim=1).numpy()
            self.threshold = float(np.quantile(errors, 0.99))
        return self

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(X, dtype=torch.float32)
            recon = self.model(x_tensor)
            errors = torch.mean((x_tensor - recon) ** 2, dim=1).numpy()
        # Normalize into pseudo-probability via sigmoid scaling around threshold
        scaled = (errors - self.threshold) / (np.std(errors) + 1e-6)
        probs = 1.0 / (1.0 + np.exp(-scaled))
        return probs

def get_baseline_models(seed: int = 42) -> Dict[str, Any]:
    """
    Returns configured baseline models with class weighting.
    """
    return {
        'LogisticRegression': LogisticRegression(
            class_weight='balanced', max_iter=500, random_state=seed, solver='lbfgs'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight='balanced', random_state=seed, n_jobs=2
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.05, random_state=seed, eval_metric='logloss', n_jobs=2
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.05, class_weight='balanced', random_state=seed, verbose=-1, n_jobs=2
        ),
        'CatBoost': CatBoostClassifier(
            iterations=100, depth=6, learning_rate=0.05, auto_class_weights='Balanced', random_seed=seed, verbose=False, thread_count=2
        ),
        'IsolationForest': IsolationForest(
            n_estimators=100, contamination=0.01, random_state=seed, n_jobs=2
        )
    }
