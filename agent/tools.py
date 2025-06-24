# tools.py
import pandas as pd
from langchain_core.tools import tool

# --- Underlying Logic Functions (These are the "real" implementations) ---

# We import your functions from the `src` directory and give them the
# standard aliases that our agent's `execute_tool` node expects.

# Step 1: Fetching (no change)
from src.fetch_data import fetch_fundamental_data as _fetch_data_logic

# Step 2: Preprocessing (no change)
from src.preprocess import preprocess_financial_data as _preprocess_data_logic

# Step 3: Prediction (NEW - using your real model)
from src.predict import predict_outperformance as _predict_performance_logic

# Step 4: Visualization (NEW - using your real plotting function)
from src.visualize import create_prediction_plot as _visualize_data_logic


# --- LangChain Tools (These definitions DO NOT CHANGE) ---
# The LLM sees these simple wrappers. The complexity of which underlying
# logic to call is handled by the `execute_tool` node in `agent.py`.
# This design makes your system robust and easy to modify.

@tool
def fetch_data(ticker: str) -> str:
    """Fetches fundamental financial data for a given stock ticker."""
    return f"[Data for {ticker} is ready to be fetched by the system.]"

@tool
def preprocess_data() -> str:
    """Preprocesses the fetched financial data. This tool takes no arguments."""
    return "[Preprocessing step is ready to be executed.]"

@tool
def predict_performance() -> str:
    """Predicts the stock's performance based on the preprocessed data. This tool takes no arguments."""
    return "[Prediction step is ready to be executed.]"

@tool
def visualize_data() -> str:
    """Visualizes the financial data and the prediction. This tool takes no arguments."""
    return "[Visualization step is ready to be executed.]"

# This list is what gets passed to the agent.
available_tools = [
    fetch_data,
    preprocess_data,
    predict_performance,
    visualize_data,
]