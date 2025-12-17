#!/usr/bin/env python
"""
Run Notebook 02 logic end-to-end outside the notebook kernel.
This script mirrors the notebook cells: setup, seed, load CSVs, DQ, feature engineering, save features and splits, write artifacts.
"""
import os
from pathlib import Path
import json
import time
import random
import numpy as np
import pandas as pd

ROOT = Path('.')
DATA_DIR = ROOT.joinpath('data','raw')
FEATURES_DIR = ROOT.joinpath('features')
ARTIFACTS = ROOT.joinpath('artifacts')
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)

DRY_RUN = True
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def write_artifact(name, obj):
    p = ARTIFACTS.joinpath(name)
    with p.open('w', encoding='utf8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print('Wrote', p)

set_seed(SEED)
print('Seeds set, DRY_RUN=', DRY_RUN)

# Load CSVs
fp_profiles = DATA_DIR.joinpath('financial_profiles_large.csv')
fp_logs = DATA_DIR.joinpath('user_intent_logs.csv')
if not fp_profiles.exists() or not fp_logs.exists():
    raise FileNotFoundError('Expected raw CSVs under data/raw. Run prototype/generate_data.py if missing.')

nrows = 1000 if DRY_RUN else None
profiles = pd.read_csv(fp_profiles, nrows=nrows)
logs = pd.read_csv(fp_logs, nrows=nrows)
print('Loaded profiles', profiles.shape, 'logs', logs.shape)

# Data quality checks
dq = {}
dq['profiles_missing'] = profiles.isnull().sum().to_dict()
dq['logs_missing'] = logs.isnull().sum().to_dict()
dq['profiles_duplicates'] = int(profiles.duplicated().sum())
dq['logs_duplicates'] = int(logs.duplicated().sum())
for col in ['loan_amount_inr', 'annual_income']:
    if col in profiles.columns:
        dq[f'{col}_negatives'] = int((profiles[col] < 0).sum())
write_artifact('dq_data_prep.json', dq)
print('DQ:', dq)

# Feature engineering
def engineer_features(profiles_df, logs_df):
    df = profiles_df.copy()
    if 'loan_amount_inr' in df.columns:
        df['loan_amount_inr'] = df['loan_amount_inr'].clip(lower=0)
        df['log_loan_amount'] = np.log1p(df['loan_amount_inr'])
    if set(['existing_debt','annual_income']).issubset(df.columns):
        df['debt_to_income'] = df['existing_debt'] / (df['annual_income'].replace(0, np.nan))
        df['debt_to_income'] = df['debt_to_income'].fillna(df['debt_to_income'].median())
    if 'user_id' in logs_df.columns and 'pause_interval' in logs_df.columns:
        agg = logs_df.groupby('user_id')['pause_interval'].agg(['mean','std','max']).rename(columns={'mean':'pause_mean','std':'pause_std','max':'pause_max'})
        df = df.merge(agg, how='left', left_on='user_id', right_index=True)
        df[['pause_mean','pause_std','pause_max']] = df[['pause_mean','pause_std','pause_max']].fillna(0)
    return df

features = engineer_features(profiles, logs)
print('features shape', features.shape)

# Save features
features_path = FEATURES_DIR.joinpath('features_table.parquet')
features.to_parquet(features_path, index=False)
print('Wrote', features_path)

sample_path = FEATURES_DIR.joinpath('features_sample.parquet')
features.sample(min(1000, len(features)), random_state=SEED).to_parquet(sample_path, index=False)
print('Wrote', sample_path)
write_artifact('features_paths.json', {'features_table': str(features_path), 'features_sample': str(sample_path)})

# Train/test split
from sklearn.model_selection import train_test_split
label_col = 'target' if 'target' in features.columns else None
if label_col is None:
    features['target'] = (features.get('credit_score', 600) < 650).astype(int)
    label_col = 'target'
train, test = train_test_split(features, test_size=0.2, random_state=SEED, stratify=features[label_col] if features[label_col].nunique()>1 else None)
train_path = ARTIFACTS.joinpath('train.parquet')
test_path = ARTIFACTS.joinpath('test.parquet')
train.to_parquet(train_path, index=False)
test.to_parquet(test_path, index=False)
write_artifact('splits.json', {'train': str(train_path), 'test': str(test_path)})
print('Wrote train/test splits')

# Feature manifest
features_manifest = {
    'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'seed': SEED,
    'features': [
        'loan_amount_inr', 'log_loan_amount', 'existing_debt', 'annual_income', 'debt_to_income',
        'credit_score', 'pause_mean', 'pause_std', 'pause_max', 'target'
    ]
}
write_artifact('features_manifest.json', features_manifest)
print('Done')
