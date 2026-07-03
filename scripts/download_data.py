import os
import pandas as pd
import numpy as np
from src.config import DATA_DIR
from src.utils.logger import logger

def get_data(n_samples: int = 2000):
    """
    Simulates downloading the dataset by generating a synthetic 'Customer Churn' dataset
    that perfectly matches the schema of Muhammad Shahid Azeem's Kaggle dataset:
    https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset
    """
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    
    if train_path.exists() and test_path.exists():
        logger.info(f"Data already exists at {train_path}. Skipping generation.")
        return

    logger.info(f"Generating synthetic Kaggle dataset with {n_samples} samples...")
    
    np.random.seed(42)
    
    # Generate synthetic features mirroring the Kaggle schema
    data = {
        'CustomerID': range(1, n_samples + 1),
        'Age': np.random.randint(18, 65, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Tenure': np.random.randint(1, 60, n_samples),
        'Usage Frequency': np.random.randint(1, 30, n_samples),
        'Support Calls': np.random.randint(0, 10, n_samples),
        'Payment Delay': np.random.randint(0, 30, n_samples),
        'Subscription Type': np.random.choice(['Basic', 'Standard', 'Premium'], n_samples),
        'Contract Length': np.random.choice(['Monthly', 'Quarterly', 'Annual'], n_samples),
        'Total Spend': np.random.uniform(100.0, 5000.0, n_samples).round(2),
        'Last Interaction': np.random.randint(1, 30, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate churn probability based on features
    churn_prob = np.zeros(n_samples)
    churn_prob += df['Support Calls'] * 0.15
    churn_prob += df['Payment Delay'] * 0.05
    churn_prob += np.where(df['Contract Length'] == 'Monthly', 0.2, -0.2)
    churn_prob -= df['Tenure'] * 0.01  
    churn_prob -= df['Usage Frequency'] * 0.02
    
    # Sigmoid to get probability
    churn_prob = 1 / (1 + np.exp(-churn_prob))
    
    # Generate binary target (1 = Churn, 0 = Stay)
    df['Churn'] = np.random.binomial(1, churn_prob)
    
    # Introduce some missing values to test feature engineer
    df.loc[np.random.choice(df.index, 50), 'Total Spend'] = np.nan
    
    # SECURITY FEATURE: PII Redaction
    # Cryptographically hash the CustomerID so no Personally Identifiable Information is exposed.
    import hashlib
    df['CustomerID'] = df['CustomerID'].apply(
        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:12]
    )
    
    # Split into train and test
    train_size = int(n_samples * 0.75)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:].drop(columns=['Churn'])
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Saved train.csv ({len(train_df)} rows) and test.csv ({len(test_df)} rows) to {DATA_DIR}")

if __name__ == "__main__":
    get_data()
