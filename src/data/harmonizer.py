"""
Harmonized Feature Layer for Network Intrusion Detection.
Implements NetFlow-v2 feature standardization, strict shortcut feature stripping
(ports, TCP window artifacts, synthetic OS fingerprints), and leak-free temporal / i.i.d. splitting.
"""

import os
import glob
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Shortcut and synthetic testbed artifact features to strictly eliminate
SHORTCUT_FEATURES = [
    'Destination Port',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'min_seg_size_forward',
    'Fwd Header Length.1',
    'Flow ID',
    'Source IP',
    'Destination IP',
    'Timestamp',
    'Source Port'
]

# Zero-variance constant features in CIC-IDS2017
ZERO_VARIANCE_FEATURES = [
    'Bwd PSH Flags',
    'Fwd URG Flags',
    'Bwd URG Flags',
    'CWE Flag Count',
    'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk',
    'Fwd Avg Bulk Rate',
    'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk',
    'Bwd Avg Bulk Rate'
]

# Standardized Attack Family Mapping
ATTACK_FAMILY_MAPPING = {
    'BENIGN': 'BENIGN',
    'FTP-Patator': 'Brute-Force',
    'SSH-Patator': 'Brute-Force',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS',
    'Heartbleed': 'DoS',
    'Web Attack  Brute Force': 'Web-Attack',
    'Web Attack - Brute Force': 'Web-Attack',
    'Web Attack  XSS': 'Web-Attack',
    'Web Attack - XSS': 'Web-Attack',
    'Web Attack  Sql Injection': 'Web-Attack',
    'Web Attack - Sql Injection': 'Web-Attack',
    'Infiltration': 'Infiltration',
    'Bot': 'Bot',
    'PortScan': 'PortScan',
    'DDoS': 'DDoS'
}

class HarmonizedDataLoader:
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.feature_names: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.imputer: Optional[SimpleImputer] = None

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.strip() for c in df.columns]
        return df

    def standardize_labels(self, label_series: pd.Series) -> pd.Series:
        # Normalize encoding artifacts
        cleaned = label_series.astype(str).str.strip()
        cleaned = cleaned.replace({
            r'Web Attack \ufffd Brute Force': 'Web-Attack',
            r'Web Attack \ufffd XSS': 'Web-Attack',
            r'Web Attack \ufffd Sql Injection': 'Web-Attack'
        }, regex=True)
        # Apply standard mapping
        return cleaned.map(lambda x: ATTACK_FAMILY_MAPPING.get(x, x if x in ATTACK_FAMILY_MAPPING.values() else 'Other-Attack'))

    def _stratified_sample(self, df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
        if len(df) <= n:
            return df
        if 'Label' not in df.columns:
            return df.sample(n=n, random_state=seed).reset_index(drop=True)
        
        sampled_indices = []
        for _, group_df in df.groupby('Label'):
            group_n = max(1, int(round(n * len(group_df) / len(df))))
            sampled_indices.extend(group_df.sample(n=min(len(group_df), group_n), random_state=seed).index.tolist())
        
        if len(sampled_indices) > n:
            rng = np.random.RandomState(seed)
            sampled_indices = rng.choice(sampled_indices, size=n, replace=False).tolist()
        return df.loc[sampled_indices].reset_index(drop=True)

    def load_day_file(self, file_pattern: str, sample_n: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
        matches = glob.glob(os.path.join(self.raw_data_dir, f"*{file_pattern}*.csv"))
        if not matches:
            raise FileNotFoundError(f"No file matching pattern '{file_pattern}' in {self.raw_data_dir}")
        
        dfs = []
        for f in matches:
            df = pd.read_csv(f, low_memory=False)
            df = self.clean_column_names(df)
            dfs.append(df)
        
        combined = pd.concat(dfs, ignore_index=True)
        if sample_n:
            combined = self._stratified_sample(combined, sample_n, seed=seed)
        return combined

    def strip_shortcuts_and_invariants(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [c for c in SHORTCUT_FEATURES + ZERO_VARIANCE_FEATURES if c in df.columns]
        return df.drop(columns=drop_cols, errors='ignore')

    def load_temporal_splits(
        self,
        train_samples: int = 60000,
        cal_samples: int = 20000,
        test_samples: int = 30000,
        seed: int = 42
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Loads realistic temporal splits across days:
        - Train: Tuesday (Brute-Force) + Wednesday (DoS)
        - Calibration/Validation: Monday (Benign baseline) + Thursday (Web-Attack / Infiltration)
        - Test (Temporal Shift & Unseen attacks): Friday (Botnet, PortScan, DDoS)
        
        Returns:
            Dict containing X_train, y_train_binary, y_train_multi, etc.
        """
        print(f"[Harmonizer] Loading temporal cross-day dataset (Seed={seed})...")
        
        # 1. Train Days: Tuesday & Wednesday
        df_tue = self.load_day_file("Tuesday", sample_n=train_samples // 2, seed=seed)
        df_wed = self.load_day_file("Wednesday", sample_n=train_samples // 2, seed=seed)
        df_train = pd.concat([df_tue, df_wed], ignore_index=True)

        # 2. Calibration / Validation Days: Monday (Pure Benign) & Thursday (Web Attacks)
        df_mon = self.load_day_file("Monday", sample_n=cal_samples // 2, seed=seed)
        df_thu_web = self.load_day_file("Thursday-WorkingHours-Morning-WebAttacks", sample_n=cal_samples // 4, seed=seed)
        df_thu_inf = self.load_day_file("Thursday-WorkingHours-Afternoon-Infilteration", sample_n=cal_samples // 4, seed=seed)
        df_cal = pd.concat([df_mon, df_thu_web, df_thu_inf], ignore_index=True)

        # 3. Test Day: Friday (Botnet, PortScan, DDoS) - Real Temporal Concept & Attack Shift
        df_fri_ddos = self.load_day_file("Friday-WorkingHours-Afternoon-DDos", sample_n=test_samples // 3, seed=seed)
        df_fri_port = self.load_day_file("Friday-WorkingHours-Afternoon-PortScan", sample_n=test_samples // 3, seed=seed)
        df_fri_bot = self.load_day_file("Friday-WorkingHours-Morning", sample_n=test_samples // 3, seed=seed)
        df_test = pd.concat([df_fri_ddos, df_fri_port, df_fri_bot], ignore_index=True)

        # Preprocessing & cleaning
        def process_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
            df = self.clean_column_names(df)
            y_multi = self.standardize_labels(df['Label'])
            y_binary = (y_multi != 'BENIGN').astype(int)
            X = df.drop(columns=['Label'], errors='ignore')
            X = self.strip_shortcuts_and_invariants(X)
            # Replace inf/-inf
            X = X.replace([np.inf, -np.inf], np.nan)
            return X, y_binary, y_multi

        X_train_df, y_train_bin, y_train_mul = process_df(df_train)
        X_cal_df, y_cal_bin, y_cal_mul = process_df(df_cal)
        X_test_df, y_test_bin, y_test_mul = process_df(df_test)

        # Harmonize common columns across splits
        common_cols = sorted(list(set(X_train_df.columns).intersection(set(X_cal_df.columns)).intersection(set(X_test_df.columns))))
        self.feature_names = common_cols
        
        X_train_df = X_train_df[common_cols]
        X_cal_df = X_cal_df[common_cols]
        X_test_df = X_test_df[common_cols]

        print(f"[Harmonizer] Total Harmonized Features (shortcuts removed): {len(common_cols)}")
        print(f"[Harmonizer] Train size: {len(X_train_df):,} | Cal size: {len(X_cal_df):,} | Test size: {len(X_test_df):,}")

        # Leak-Free Imputation & Standardization (fit strictly on Train!)
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()

        X_train = self.scaler.fit_transform(self.imputer.fit_transform(X_train_df))
        X_cal = self.scaler.transform(self.imputer.transform(X_cal_df))
        X_test = self.scaler.transform(self.imputer.transform(X_test_df))

        return {
            'X_train': X_train,
            'y_train_binary': y_train_bin.values,
            'y_train_multi': y_train_mul.values,
            'X_cal': X_cal,
            'y_cal_binary': y_cal_bin.values,
            'y_cal_multi': y_cal_mul.values,
            'X_test': X_test,
            'y_test_binary': y_test_bin.values,
            'y_test_multi': y_test_mul.values,
            'feature_names': common_cols,
            'train_df_raw': X_train_df,
            'test_df_raw': X_test_df
        }
