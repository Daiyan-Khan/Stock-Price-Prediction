import os
import requests
import pandas as pd
from api.data_service import get_stock_list

API_KEY = "UE4CU1RV2HT3L1UI"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
DATA_FOLDER = "data\individual_stocks_5yr"  # Specify the directory for storing CSV files

def fetch_latest_data(stock_symbol):
    """Fetch the latest stock data from Alpha Vantage API."""
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": stock_symbol,
        "interval": "60min",
        "apikey": API_KEY,
        "outputsize": "compact"
    }
    
    response = requests.get(ALPHA_VANTAGE_URL, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to fetch data for {stock_symbol}. Status code: {response.status_code}")

def preprocess_data(stock_symbol, json_data):
    """Preprocess the stock data and return as DataFrame."""
    time_series = json_data.get('Time Series (60min)', {})
    data_list = []
    
    for timestamp, values in time_series.items():
        data_list.append({
            "date": timestamp,
            "open": float(values["1. open"]),
            "high": float(values["2. high"]),
            "low": float(values["3. low"]),
            "close": float(values["4. close"]),
            "volume": int(values["5. volume"]),
            "stock_name": stock_symbol
        })
    
    return pd.DataFrame(data_list)

def update_all_stocks():
    """Fetch and preprocess data for all stocks."""
    stock_symbols = get_stock_list()
    
    for stock in stock_symbols:
        try:
            json_data = fetch_latest_data(stock)
            new_data = preprocess_data(stock, json_data)
            
            # Define the CSV file path
            csv_file_path = os.path.join(DATA_FOLDER, f"{stock}.csv")
            
            # Check if the CSV file exists
            if os.path.exists(csv_file_path):
                # Load existing data
                existing_data = pd.read_csv(csv_file_path)
                # Append the new data to the existing data
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            else:
                # If the file doesn't exist, use the new data as the combined data
                combined_data = new_data
            
            # Save the combined data back to CSV
            combined_data.to_csv(csv_file_path, index=False)
            
            # Optionally, you can call the save_preprocessed_data function if needed
            # save_preprocessed_data(stock, combined_data)

        except Exception as e:
            print(f"Error updating {stock}: {str(e)}")

# Example function call (uncomment to run)
# update_all_stocks()
