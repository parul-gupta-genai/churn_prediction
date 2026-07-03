import pandas as pd
from src.config import DATA_DIR
from src.utils.logger import logger
from scripts.download_data import get_data
from src.agents.graph import build_pipeline_graph

def main():
    logger.info("Starting Customer Churn Pipeline...")
    
    # 1. Ensure we have data (generates synthetic data matching Kaggle schema)
    get_data()
    
    # 2. Build and run the graph
    app = build_pipeline_graph()
    
    # Initial state
    initial_state = {}
    
    logger.info("Executing pipeline graph...")
    final_state = app.invoke(initial_state)
    
    # 3. Save Submission
    test_path = DATA_DIR / "test.csv"
    test_df = pd.read_csv(test_path)
    
    submission = pd.DataFrame({
        'CustomerID': test_df['CustomerID'],
        'Churn': final_state['ensemble_preds'] # Outputting probabilities
    })
    
    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)
    logger.info(f"Pipeline complete! Submission saved to {submission_path}")

if __name__ == "__main__":
    main()
