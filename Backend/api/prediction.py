from flask import Flask, request, jsonify
import os
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_service import load_data
import logging
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Define directories
model_save_dir = 'model\\trained_models'
data_dir = 'data\pre_processed_data'

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.get_json()

    # Debug: Check the incoming data
    app.logger.debug(f"Received data: {data}")

    # Validate input
    if not data or 'stock_name' not in data:
        app.logger.error("Invalid input: 'stock_name' is missing")
        return jsonify({'error': 'Invalid input'}), 400

    stock_name = data['stock_name']
    model_path = os.path.join(model_save_dir, f"{stock_name}_preprocessed_data_lstm_model.keras")

    # Debug: Check the model path
    app.logger.debug(f"Model path: {model_path}")

    # Check if the model exists
    if not os.path.exists(model_path):
        app.logger.error(f"Model for {stock_name} not found at {model_path}")
        return jsonify({'error': f'Model for {stock_name} not found'}), 404

    # Load the trained LSTM model
    try:
        model = tf.keras.models.load_model(model_path)
        app.logger.debug(f"Model for {stock_name} loaded successfully")
    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}")
        return jsonify({'error': str(e)}), 500

    # Load pre-processed data for prediction
    csv_file = os.path.join(data_dir, f"{stock_name}_preprocessed_data.csv")
    try:
        # Load the CSV data and select only the 'close' prices for predictions
        app.logger.debug(f"Loading data from: {csv_file}")
        df = pd.read_csv(csv_file, usecols=['date', 'close'], index_col='date', parse_dates=True)
        close_data = df[['close']].values
        
        # Preprocess using MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_data)

        # Use the last 10 time steps for prediction
        time_steps = 10
        last_data = scaled_data[-time_steps:].reshape(1, time_steps, 1)
        app.logger.debug("Data preprocessing successful")
    except Exception as e:
        app.logger.error(f"Error loading or processing data: {str(e)}")
        return jsonify({'error': str(e)}), 500  # Handle data loading errors

    # Make prediction using the LSTM model
    try:
        predicted_price = model.predict(last_data)
        app.logger.debug(f"Prediction successful: {predicted_price}")
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

    # Inverse transform the predicted price to get the actual price
    predicted_price = scaler.inverse_transform(predicted_price)[0][0]

    predicted_price = float(predicted_price)  # Convert from float32 to Python float
    return jsonify({'predicted_price': predicted_price}), 200


if __name__ == "__main__":
    # Debug: Ensure app runs in debug mode
    app.run(host='0.0.0.0', port=5001, debug=True)
