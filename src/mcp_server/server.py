import asyncio
from mcp.server import Server
import mcp.types as types
import pandas as pd
import joblib
from pathlib import Path
import json

from src.features.engineer import FeatureEngineer
from src.config import MODELS_DIR

# Initialize MCP Server
app = Server("churn_prediction_mcp")

# Load models at startup
@app.lifespan
async def lifespan(server: Server):
    print("Loading Churn Prediction Models for MCP...")
    # Global state for models
    global fe, xgb_model, lgb_model
    train_df = pd.read_csv("data/train.csv")
    fe = FeatureEngineer()
    fe.fit_transform(train_df)
    xgb_model = joblib.load(MODELS_DIR / "xgb_model.pkl")
    lgb_model = joblib.load(MODELS_DIR / "lgb_model.pkl")
    yield
    print("Shutting down MCP server.")

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    """Expose the churn prediction model as an AI tool."""
    return [
        types.Tool(
            name="predict_customer_churn",
            description="Predicts the probability of a customer churning based on their profile.",
            inputSchema={
                "type": "object",
                "properties": {
                    "age": {"type": "integer", "description": "Age of the customer"},
                    "gender": {"type": "string", "enum": ["Male", "Female"]},
                    "tenure": {"type": "integer", "description": "Months with the company"},
                    "usage_frequency": {"type": "integer", "description": "Service usage frequency (1-30)"},
                    "support_calls": {"type": "integer", "description": "Number of support calls made"},
                    "payment_delay": {"type": "integer", "description": "Days payment is delayed"},
                    "subscription_type": {"type": "string", "enum": ["Basic", "Standard", "Premium"]},
                    "contract_length": {"type": "string", "enum": ["Monthly", "Quarterly", "Annual"]},
                    "total_spend": {"type": "number", "description": "Total amount spent ($)"},
                    "last_interaction": {"type": "integer", "description": "Days since last interaction"}
                },
                "required": ["age", "gender", "tenure", "usage_frequency", "support_calls", "payment_delay", "subscription_type", "contract_length", "total_spend", "last_interaction"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Execute the tool when called by an AI client."""
    if name != "predict_customer_churn":
        raise ValueError(f"Unknown tool: {name}")
        
    try:
        # Construct DataFrame
        input_data = {
            'CustomerID': [99999], # Dummy ID
            'Age': [arguments['age']],
            'Gender': [arguments['gender']],
            'Tenure': [arguments['tenure']],
            'Usage Frequency': [arguments['usage_frequency']],
            'Support Calls': [arguments['support_calls']],
            'Payment Delay': [arguments['payment_delay']],
            'Subscription Type': [arguments['subscription_type']],
            'Contract Length': [arguments['contract_length']],
            'Total Spend': [arguments['total_spend']],
            'Last Interaction': [arguments['last_interaction']]
        }
        df = pd.DataFrame(input_data)
        
        # Transform & Predict
        X = fe.transform(df)
        xgb_prob = xgb_model.predict_proba(X)[0][1]
        lgb_prob = lgb_model.predict_proba(X)[0][1]
        ensemble_prob = float((xgb_prob + lgb_prob) / 2.0)
        
        result = {
            "churn_probability": ensemble_prob,
            "risk_level": "High" if ensemble_prob > 0.5 else "Low",
            "recommendation": "Offer discount or priority support" if ensemble_prob > 0.5 else "No action needed"
        }
        
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
        
    except Exception as e:
        return [types.TextContent(type="text", text=f"Error running prediction: {str(e)}")]

if __name__ == "__main__":
    from mcp.server.stdio import stdio_server
    
    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
            
    asyncio.run(run())
