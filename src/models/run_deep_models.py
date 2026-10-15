"""
Isolated Runner for Deep Tabular & Anomaly PyTorch Models.
Runs in a clean subprocess to prevent macOS OpenMP runtime conflicts with TreeSHAP/LightGBM.
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.deep_tabular import PyTorchTabularClassifier
from src.models.baselines import AutoencoderAnomalyDetector
from pytorch_tabnet.tab_model import TabNetClassifier
from src.evaluation.metrics import evaluate_intrusion_detector

def run_deep_models(data_path: str, output_path: str, seed: int = 42):
    data = np.load(data_path)
    X_train = data['X_train']
    y_train_bin = data['y_train_binary']
    X_test = data['X_test']
    y_test_bin = data['y_test_binary']
    in_dim = X_train.shape[1]

    results = {}

    # 1. Deep Autoencoder (trained on benign flows)
    try:
        t0 = time.perf_counter()
        ae = AutoencoderAnomalyDetector(input_dim=in_dim, epochs=8, seed=seed)
        ae.fit(X_train[y_train_bin == 0])
        ae_scores = ae.predict_anomaly_score(X_test)
        lat = ((time.perf_counter() - t0) / len(X_test)) * 1e6
        results['DeepAutoencoder'] = evaluate_intrusion_detector(y_test_bin, ae_scores, latency_per_sample_us=lat)
    except Exception as e:
        print(f"Error DeepAutoencoder: {e}")

    # 2. PyTorch Tabular Architectures
    for dt_type, dt_key in [('mlp', 'Deep-MLP'), ('cnn', 'Deep-1DCNN'), ('lstm', 'Deep-LSTM'), ('ft_transformer', 'Deep-FTTransformer')]:
        try:
            t0 = time.perf_counter()
            nn_clf = PyTorchTabularClassifier(model_type=dt_type, in_features=in_dim, epochs=6, seed=seed)
            nn_clf.fit(X_train, y_train_bin)
            t_infer = time.perf_counter()
            p_nn = nn_clf.predict_proba(X_test)[:, 1]
            lat = ((time.perf_counter() - t_infer) / len(X_test)) * 1e6
            results[dt_key] = evaluate_intrusion_detector(y_test_bin, p_nn, latency_per_sample_us=lat)
        except Exception as e:
            print(f"Error {dt_key}: {e}")

    # 3. TabNet
    try:
        t0 = time.perf_counter()
        tabnet = TabNetClassifier(verbose=0, seed=seed)
        tabnet.fit(X_train, y_train_bin, max_epochs=6, batch_size=256, virtual_batch_size=64)
        t_infer = time.perf_counter()
        p_tab = tabnet.predict_proba(X_test)[:, 1]
        lat = ((time.perf_counter() - t_infer) / len(X_test)) * 1e6
        results['TabNet'] = evaluate_intrusion_detector(y_test_bin, p_tab, latency_per_sample_us=lat)
    except Exception as e:
        print(f"Error TabNet: {e}")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[Deep Tabular] Evaluated {len(results)} architectures successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_deep_models(args.data_path, args.output_path, args.seed)
