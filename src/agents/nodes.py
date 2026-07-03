from src.agents.state import PipelineState
from src.data_layer.loader import load_data
from src.eda.visualizer import generate_eda_plots
from src.features.engineer import FeatureEngineer
from src.models.sklearn_models import train_xgboost, train_lightgbm
from src.models.torch_model import train_pytorch_mlp
from src.models.ensemble import EnsembleBlender
from src.models.registry import save_model
from src.utils.logger import logger

def load_data_node(state: PipelineState) -> PipelineState:
    logger.info("--- Node: Load Data ---")
    train_df, test_df = load_data()
    return {"train_df": train_df, "test_df": test_df}

def eda_node(state: PipelineState) -> PipelineState:
    logger.info("--- Node: EDA ---")
    generate_eda_plots(state["train_df"])
    return state

def feature_engineer_node(state: PipelineState) -> PipelineState:
    logger.info("--- Node: Feature Engineering ---")
    fe = FeatureEngineer()
    X_train, y_train = fe.fit_transform(state["train_df"])
    X_test = fe.transform(state["test_df"])
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "feature_engineer": fe
    }

def train_node(state: PipelineState) -> PipelineState:
    logger.info("--- Node: Training Models ---")
    X_train = state["X_train"]
    y_train = state["y_train"]
    
    xgb_model = train_xgboost(X_train, y_train)
    lgb_model = train_lightgbm(X_train, y_train)
    torch_model = train_pytorch_mlp(X_train, y_train)
    
    save_model(xgb_model, "xgb_model")
    save_model(lgb_model, "lgb_model")
    save_model(torch_model, "torch_model")
    
    return {
        "models": {
            "xgb": xgb_model,
            "lgb": lgb_model,
            "torch": torch_model
        }
    }

def ensemble_node(state: PipelineState) -> PipelineState:
    logger.info("--- Node: Ensembling ---")
    
    models = [
        state["models"]["xgb"], 
        state["models"]["lgb"], 
        state["models"]["torch"]
    ]
    
    blender = EnsembleBlender(models, weights=[0.4, 0.4, 0.2])
    preds = blender.predict(state["X_test"])
    
    return {"ensemble_preds": preds}
