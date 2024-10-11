from flask import Flask, request, jsonify
import json
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from flask_cors import CORS  # Import CORS
import requests

# Create a Flask application
app = Flask(__name__)
CORS(app)

# Define directories
preprocessed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pre_processed_data')
predictions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','predictions')
os.makedirs(preprocessed_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)

# Define your functions here
def load_data(file_path):
    try:
        return pd.read_csv(file_path)  # Make sure this returns a DataFrame
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None  # Or handle it in a way that suits your needs

def create_training_dataset(data, look_back=60):
    """Function to create the training dataset for the model."""
    X_train, y_train = [], []
    for i in range(look_back, len(data)):
        X_train.append(data[i - look_back:i])
        y_train.append(data[i])
    
    # Convert to NumPy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshape X_train for LSTM input (samples, time steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train

def preprocess_and_store(stock_symbol, json_data):
    """Preprocess stock data and store it in the preprocessed_data folder."""
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
    
    # Convert to DataFrame
    new_data = pd.DataFrame(data_list)
    
    # Append to the existing CSV file
    file_path = os.path.join(preprocessed_dir, f"{stock_symbol}_preprocessed_data.csv")
    if os.path.exists(file_path):
        existing_data = pd.read_csv(file_path)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    else:
        updated_data = new_data
    
    updated_data.to_csv(file_path, index=False)
    print(f"Data for {stock_symbol} has been updated and preprocessed.")

# Define the API endpoint to get preprocessed stock data
@app.route('/api/stocks', methods=['GET'])
def api_get_stocks():
    """API endpoint to retrieve preprocessed stock data."""
    data_dir = 'data/pre_processed_data'
    
    if not os.path.exists(data_dir):
        return jsonify({"error": f"Data directory {data_dir} does not exist."}), 404

    # List all CSV files in the directory
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    stocks_data = {}
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        data = load_data(file_path)
        stocks_data[file] = data['close'].tolist()

    return jsonify(stocks_data), 200

@app.route('/api/stocks/<stock_name>/details', methods=['GET'])
def api_get_stock_details(stock_name):
    """API endpoint to retrieve actual and predicted data for a specific stock."""
    data_dir = 'data/pre_processed_data'
    predictions_dir = 'data/predictions'  # Ensure this directory is defined
    print(f"Received request for stock details: {stock_name}")

    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return jsonify({"error": f"Data directory {data_dir} does not exist."}), 404

    # Construct file path for the stock data
    file_path = os.path.join(data_dir, f"{stock_name}_preprocessed_data.csv")
    print(f"Looking for file: {file_path}")

    if not os.path.exists(file_path):
        print(f"Stock data for {stock_name} does not exist.")
        return jsonify({"error": f"Stock data for {stock_name} does not exist."}), 404

    # Load actual stock data
    actual_data = load_data(file_path)

    # Load predictions from the new endpoint
    predictions_file_path = os.path.join(predictions_dir, f"{stock_name}_preprocessed_data_predictions.csv")
    print(f"Looking for predictions file: {predictions_file_path}")
    if os.path.exists(predictions_file_path):
        predictions_data = pd.read_csv(predictions_file_path)

        # Get the last 10 predictions
        last_10_predictions = predictions_data.tail(10)[['date', 'predicted_price']].to_dict(orient='records')
        next_day_prediction = predictions_data.iloc[-1]['predicted_price'] if not predictions_data.empty else None
    else:
        print(f"No predictions found for {stock_name}.")
        last_10_predictions = []
        next_day_prediction = None

    # Combine actual and predicted data
    result = {
        "actual": actual_data.to_dict(orient='records'),
        "predicted": last_10_predictions,
        "next_day_prediction": next_day_prediction
    }

    return jsonify(result), 200



def get_stock_list():
    """Function to retrieve the list of stock symbols from the data directory."""
    data_dir = 'data/individual_stocks_5yr/individual_stocks_5yr'
    
    # Check if the directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return []

    # List all CSV files in the directory
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    # Extract stock symbols from the filenames
    stock_symbols = [os.path.splitext(file)[0] for file in files]
    
    return stock_symbols

@app.route('/api/predictions/<stock_name>', methods=['GET'])
def load_predictions(stock_name):
    """API endpoint to load predictions for the last 10 days and the next day."""
    predictions_file = os.path.join(predictions_dir, f"{stock_name}_preprocessed_data_predictions.csv")

    if not os.path.exists(predictions_file):
        return jsonify({"error": f"No predictions found for {stock_name}."}), 404

    predictions_data = pd.read_csv(predictions_file)

    # Get the last 10 days' predictions and the next day's prediction
    last_10_days = predictions_data.tail(10).to_dict(orient='records')
    next_day_prediction = predictions_data.iloc[-1].to_dict()

    return jsonify({
        "last_10_days": last_10_days,
        "next_day_prediction": next_day_prediction
    }), 200

# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
