from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict, Any
from scripts.download_data import get_data
from src.agents.graph import build_pipeline_graph
from src.utils.logger import logger
import numpy as np

app = FastAPI(
    title="Customer Churn Pipeline API",
    description="Run the end-to-end ML Pipeline via REST API with Dataset controls and Security Features",
    version="1.2.0"
)

# SECURITY FEATURE: API Authentication
API_KEY = "capstone-secret-key"
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials")

class PipelineResponse(BaseModel):
    status: str
    message: str
    preview_predictions: List[Dict[str, Any]]

class DatasetOptions(BaseModel):
    n_samples: int = 2000

@app.get("/")
def read_root():
    return {"message": "Welcome to the Customer Churn Prediction API! Go to /docs for the swagger UI."}

@app.get("/dataset/info", dependencies=[Depends(get_api_key)])
def dataset_info():
    """Returns information about the current dataset in the data folder."""
    try:
        train_df = pd.read_csv("data/train.csv")
        test_df = pd.read_csv("data/test.csv")
        return {
            "status": "success",
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "features": list(train_df.columns)
        }
    except FileNotFoundError:
        return {"status": "missing", "message": "Dataset not found. Run /dataset/generate first."}

@app.post("/dataset/generate", dependencies=[Depends(get_api_key)])
def generate_dataset(options: DatasetOptions):
    """Generates a fresh synthetic dataset with a specific number of samples."""
    logger.info(f"API: Generating new dataset with {options.n_samples} samples...")
    # Temporarily override the hardcoded 2000 in download_data by hacking it or just calling the script
    # For simplicity, we just delete old files and call get_data() which generates 2000 if missing.
    # But to make options work, let's just write a custom generator here or update the script.
    
    # We will just call get_data() for now, which ensures it exists. 
    # To truly use options.n_samples, download_data.py needs an update. 
    import os
    if os.path.exists("data/train.csv"):
        os.remove("data/train.csv")
    if os.path.exists("data/test.csv"):
        os.remove("data/test.csv")
        
    get_data(options.n_samples)
    return {"status": "success", "message": f"Generated new dataset with {options.n_samples} samples."}

@app.post("/run-pipeline", response_model=PipelineResponse)
def run_pipeline():
    """Triggers the entire data loading, training, and ensembling pipeline."""
    try:
        logger.info("API: Triggering Pipeline Run...")
        
        get_data()
        
        pipeline_app = build_pipeline_graph()
        final_state = pipeline_app.invoke({})
        
        test_path = "data/test.csv"
        test_df = pd.read_csv(test_path)
        
        submission = pd.DataFrame({
            'CustomerID': test_df['CustomerID'],
            'Churn_Probability': final_state['ensemble_preds']
        })
        
        submission_path = "submission.csv"
        submission.to_csv(submission_path, index=False)
        
        # Get top 5 predictions to show as results
        preview = submission.head(5).to_dict(orient="records")
        
        logger.info("API: Pipeline Run Complete.")
        return PipelineResponse(
            status="success", 
            message="Pipeline executed successfully. Full results saved to submission.csv",
            preview_predictions=preview
        )
    except Exception as e:
        logger.error(f"API: Pipeline Run Failed - {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
