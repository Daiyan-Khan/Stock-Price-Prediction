from flask import Flask, request, jsonify
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from flask_cors import CORS  # Import CORS
# Create a Flask application
app = Flask(__name__)
CORS(app)
# Define your functions here
def load_data(filepath):
    """Function to load data from a file."""
    return pd.read_csv(filepath)

def create_training_dataset(data, look_back=60):
    """Function to create the training dataset for the model."""
    X_train, y_train = [], []
    for i in range(look_back, len(data)):
        X_train.append(data[i-look_back:i])
        y_train.append(data[i])
    
    # Convert to NumPy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshape X_train for LSTM input (samples, time steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train

def save_model(model, filepath):
    """Function to save the trained model."""
    model.save(filepath)
    print(f"Model saved at {filepath}")

def train_model(stock_data_filepath, model_save_path):
    """Function to train the LSTM model."""
    # Load the data
    data = load_data(stock_data_filepath)
    
    # Assuming the data is in a column named 'close'
    close_prices = data['close'].values
    
    # Normalize the dataset (for better model performance)
    close_prices_scaled = close_prices / np.max(close_prices)

    # Create the training dataset
    X_train, y_train = create_training_dataset(close_prices_scaled)

    # Create the LSTM model
    model = tf.keras.Sequential()
    model.add(tf.keras.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(tf.keras.Dropout(0.2))
    model.add(tf.keras.LSTM(units=50, return_sequences=False))
    model.add(tf.keras.Dropout(0.2))
    model.add(tf.keras.Dense(units=25))
    model.add(tf.keras.Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1)

    # Save the model
    save_model(model, model_save_path)

# Define the API endpoint for training the model
@app.route('/api/train', methods=['POST'])
def api_train_model():
    """API endpoint to train the model."""
    stock_file = request.json.get('stock_file')  # Expecting JSON payload
    model_save_path = request.json.get('model_save_path')  # Expecting JSON payload

    if not stock_file or not model_save_path:
        return jsonify({"error": "Please provide 'stock_file' and 'model_save_path'."}), 400
    
    # Set the directory for stock data
    data_dir = 'data/individual_stocks_5yr/individual_stocks_5y'
    stock_data_filepath = os.path.join(data_dir, stock_file)

    if not os.path.exists(stock_data_filepath):
        return jsonify({"error": f"File {stock_file} does not exist in {data_dir}."}), 404

    try:
        train_model(stock_data_filepath, model_save_path)
        return jsonify({"message": "Model training completed successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Define the API endpoint to get preprocessed stock data
from flask import Flask, request, jsonify
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from flask_cors import CORS  # Import CORS
# Create a Flask application
app = Flask(__name__)
CORS(app)
# Define your functions here
def load_data(filepath):
    """Function to load data from a file."""
    return pd.read_csv(filepath)

def create_training_dataset(data, look_back=60):
    """Function to create the training dataset for the model."""
    X_train, y_train = [], []
    for i in range(look_back, len(data)):
        X_train.append(data[i-look_back:i])
        y_train.append(data[i])
    
    # Convert to NumPy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshape X_train for LSTM input (samples, time steps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    return X_train, y_train

def save_model(model, filepath):
    """Function to save the trained model."""
    model.save(filepath)
    print(f"Model saved at {filepath}")

def train_model(stock_data_filepath, model_save_path):
    """Function to train the LSTM model."""
    # Load the data
    data = load_data(stock_data_filepath)
    
    # Assuming the data is in a column named 'close'
    close_prices = data['close'].values
    
    # Normalize the dataset (for better model performance)
    close_prices_scaled = close_prices / np.max(close_prices)

    # Create the training dataset
    X_train, y_train = create_training_dataset(close_prices_scaled)

    # Create the LSTM model
    model = tf.keras.Sequential()
    model.add(tf.keras.LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(tf.keras.Dropout(0.2))
    model.add(tf.keras.LSTM(units=50, return_sequences=False))
    model.add(tf.keras.Dropout(0.2))
    model.add(tf.keras.Dense(units=25))
    model.add(tf.keras.Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1)

    # Save the model
    save_model(model, model_save_path)

# Define the API endpoint for training the model
@app.route('/api/train', methods=['POST'])
def api_train_model():
    """API endpoint to train the model."""
    stock_file = request.json.get('stock_file')  # Expecting JSON payload
    model_save_path = request.json.get('model_save_path')  # Expecting JSON payload

    if not stock_file or not model_save_path:
        return jsonify({"error": "Please provide 'stock_file' and 'model_save_path'."}), 400
    
    # Set the directory for stock data
    data_dir = 'data/individual_stocks_5yr/individual_stocks_5y'
    stock_data_filepath = os.path.join(data_dir, stock_file)

    if not os.path.exists(stock_data_filepath):
        return jsonify({"error": f"File {stock_file} does not exist in {data_dir}."}), 404

    try:
        train_model(stock_data_filepath, model_save_path)
        return jsonify({"message": "Model training completed successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Define the API endpoint to get preprocessed stock data
@app.route('/api/stocks', methods=['GET'])
def api_get_stocks():
    """API endpoint to retrieve preprocessed stock data."""
    data_dir = 'data\pre_processed_data'
    
    if not os.path.exists(data_dir):
        return jsonify({"error": f"Data directory {data_dir} does not exist."}), 404

    # List all CSV files in the directory
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    stocks_data = {}
    
    for file in files:
        file_path = os.path.join(data_dir, file)
        data = load_data(file_path)
        # Assuming you want to send back the 'close' prices or any specific column
        stocks_data[file] = data['close'].tolist()

    return jsonify(stocks_data), 200
@app.route('/api/stocks/<stock_name>', methods=['GET'])
def api_get_stock(stock_name):
    """API endpoint to retrieve the latest data for a specific stock."""
    data_dir = 'data\pre_processed_data'
    print(f"Received request for stock: {stock_name}")

    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist.")
        return jsonify({"error": f"Data directory {data_dir} does not exist."}), 404

    # Construct file path for the stock data
    file_path = os.path.join(data_dir, f"{stock_name}_preprocessed_data.csv")
    print(f"Looking for file: {file_path}")

    if not os.path.exists(file_path):
        print(f"Stock data for {stock_name} does not exist.")
        return jsonify({"error": f"Stock data for {stock_name} does not exist."}), 404

    data = load_data(file_path)
    return jsonify(data.to_dict(orient='records')), 200


# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)


# Run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003)
