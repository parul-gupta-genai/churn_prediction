from typing import TypedDict, Any, Dict

class PipelineState(TypedDict):
    """Represents the state of our Customer Churn pipeline."""
    # Data
    train_df: Any
    test_df: Any
    
    # Processed Data
    X_train: Any
    y_train: Any
    X_test: Any
    
    # Feature Engineer
    feature_engineer: Any
    
    # Models
    models: Dict[str, Any]
    
    # Predictions
    ensemble_preds: Any
