# src/visualize.py

#### Fonction dummy pour les premiers tests #### 
import pandas as pd
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO

def create_prediction_plot(ticker: str, data: pd.DataFrame, prediction: str) -> str:
    """
    Creates a visualization of the financial data and the prediction for a given ticker.
    Returns the base64 encoded string of the generated image.
    """
    print(f"Creating visualization for {ticker} with prediction: {prediction}...")

    if data.empty:
        raise ValueError("No data to visualize.")

    # Select some key metrics for visualization. Adjust based on your data.
    metrics_to_plot = ['Revenue', 'Net Income', 'EPS']
    plot_data = data[metrics_to_plot].iloc[-1] # Get latest year for simplicity

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_title(f'{ticker} Key Financial Metrics (Latest Year)\nPrediction: {prediction}', fontsize=14)
    ax.set_ylabel('Value (in Billions)', fontsize=12)
    ax.set_xlabel('Metric', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot to a BytesIO object and then encode to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close(fig) # Close the figure to free memory
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    print(f"Visualization created for {ticker}.")
    return image_base64

if __name__ == '__main__':
    # Example usage for testing
    from src.fetch_data import fetch_fundamental_data
    try:
        aapl_raw = fetch_fundamental_data("AAPL")
        # In a real scenario, you'd pass the actual prediction here
        base64_image = create_prediction_plot("AAPL", aapl_raw, "Outperform")
        # To verify, you could decode and save this to a file:
        # with open("test_plot.png", "wb") as f:
        #     f.write(base64.b64decode(base64_image))
        print("\nAAPL Plot created (base64 string returned).")
    except Exception as e:
        print(f"Error creating plot for AAPL: {e}")