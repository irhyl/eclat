#!/usr/bin/env python3
"""
Simple synthetic data generator for éclat prototype.
Writes:
 - data/raw/financial_profiles_large.csv
 - data/raw/user_intent_logs.csv

Usage: python prototype/generate_data.py --rows 10000
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import random

parser = argparse.ArgumentParser()
parser.add_argument('--rows', type=int, default=10000)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

ROWS = args.rows
SEED = args.seed
np.random.seed(SEED)
random.seed(SEED)

ROOT = Path('.')
DATA_DIR = ROOT.joinpath('data','raw')
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Generate financial profiles
user_ids = [f'USR_{i:05d}' for i in range(ROWS)]
# loan amounts in INR, skewed distribution
loan_amount = (np.random.lognormal(mean=10, sigma=1.0, size=ROWS) / 1000).astype(int) * 1000
# annual income around 300k-3M INR
annual_income = (np.random.lognormal(mean=12, sigma=0.8, size=ROWS) / 1000).astype(int) * 1000
existing_debt = (loan_amount * np.random.beta(1.5,4.0,size=ROWS)).astype(int)
credit_score = np.clip(np.random.normal(loc=650, scale=70, size=ROWS).astype(int), 300, 850)

profiles = pd.DataFrame({
    'user_id': user_ids,
    'loan_amount_inr': loan_amount,
    'existing_debt': existing_debt,
    'annual_income': annual_income,
    'credit_score': credit_score
})

profiles.to_csv(DATA_DIR.joinpath('financial_profiles_large.csv'), index=False)
print('Wrote', DATA_DIR.joinpath('financial_profiles_large.csv'))

# Generate user intent logs; for each user create 3-10 utterances with pause intervals
rows = []
for uid in user_ids:
    k = random.randint(3,10)
    # simulate pauses in seconds: short (~0.5s) to long (~8s)
    pauses = np.abs(np.random.normal(loc=1.0, scale=1.5, size=k))
    # intent label stub
    intents = np.random.choice(['apply_loan','ask_terms','hesitant','clarify'], size=k, p=[0.4,0.25,0.2,0.15])
    timestamps = np.cumsum(np.maximum(0.1, pauses))
    for i in range(k):
        rows.append({'user_id': uid, 'utterance_id': f'{uid}_U{i}', 'pause_interval': float(pauses[i]), 'timestamp': float(timestamps[i]), 'intent_label': intents[i]})

logs = pd.DataFrame(rows)
logs.to_csv(DATA_DIR.joinpath('user_intent_logs.csv'), index=False)
print('Wrote', DATA_DIR.joinpath('user_intent_logs.csv'))
