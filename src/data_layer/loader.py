import pandas as pd
from typing import Tuple
from src.config import DATA_DIR
from src.utils.logger import logger

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads train and test datasets."""
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    
    if not train_path.exists() or not test_path.exists():
        logger.error("Data not found. Please run scripts/download_data.py first.")
        raise FileNotFoundError("Dataset missing.")
        
    logger.info("Loading training and testing data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df
