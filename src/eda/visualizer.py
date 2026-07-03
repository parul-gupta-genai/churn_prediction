import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import PROJECT_ROOT
from src.utils.logger import logger

def generate_eda_plots(train_df: pd.DataFrame):
    """Generates basic EDA plots and saves them."""
    logger.info("Generating EDA plots...")
    
    eda_dir = PROJECT_ROOT / "reports" / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    
    # Set dark theme
    plt.style.use('dark_background')
    
    # 1. Target distribution (Churn)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=train_df, x='Churn', palette='viridis')
    plt.title('Customer Churn Distribution')
    plt.savefig(eda_dir / "target_dist.png")
    plt.close()
    
    # 2. Correlation matrix (numerical only)
    plt.figure(figsize=(10, 8))
    corr = train_df.select_dtypes(include=['number']).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Feature Correlation Matrix')
    plt.savefig(eda_dir / "corr_matrix.png")
    plt.close()
    
    logger.info(f"EDA plots saved to {eda_dir}")
