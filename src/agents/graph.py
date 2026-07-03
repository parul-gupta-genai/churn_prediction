from langgraph.graph import StateGraph, END
from src.agents.state import PipelineState
from src.agents.nodes import (
    load_data_node,
    eda_node,
    feature_engineer_node,
    train_node,
    ensemble_node
)
from src.utils.logger import logger

def build_pipeline_graph():
    logger.info("Building StateGraph pipeline...")
    
    workflow = StateGraph(PipelineState)
    
    workflow.add_node("load", load_data_node)
    workflow.add_node("eda", eda_node)
    workflow.add_node("feature", feature_engineer_node)
    workflow.add_node("train", train_node)
    workflow.add_node("ensemble", ensemble_node)
    
    workflow.set_entry_point("load")
    
    workflow.add_edge("load", "eda")
    workflow.add_edge("eda", "feature")
    workflow.add_edge("feature", "train")
    workflow.add_edge("train", "ensemble")
    workflow.add_edge("ensemble", END)
    
    app = workflow.compile()
    return app
