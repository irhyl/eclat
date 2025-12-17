import pandas as pd

for f in ['data/raw/financial_profiles_large.csv','data/raw/user_intent_logs.csv']:
    try:
        df = pd.read_csv(f, nrows=3)
        print(f"{f}: ok shape={df.shape}")
        print(df.head().to_string())
    except Exception as e:
        print(f"{f}: error: {e}")
