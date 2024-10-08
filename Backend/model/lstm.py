import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import multiprocessing

# Function to load and preprocess data from CSV
def load_data(csv_file):
    # Load the CSV file
    data = pd.read_csv(csv_file)
    
    # Extract relevant columns (e.g., 'close' price)
    prices = data['close'].values
    prices = prices.reshape(-1, 1)  # Reshape for scaling

    # Scale the data to [0, 1] range
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices)

    return scaled_data, scaler

# Function to create datasets
def create_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps - 1):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# Function to build the LSTM model
def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True, stateful=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128, return_sequences=False, stateful=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to extract weights, biases, and recurrent weights
def extract_weights_biases(model):
    lstm_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.LSTM)]
    weights_data = []

    for lstm_layer in lstm_layers:
        W, U, B = lstm_layer.get_weights()  # W = input kernel, U = recurrent kernel, B = bias
        weights_data.append({
            "W": W.tolist(),  # Converting numpy arrays to list for saving in CSV
            "U": U.tolist(),
            "B": B.tolist()
        })
    
    return weights_data

# Function to train the model for a given stock
def train_model_for_stock(csv_files):
    time_steps = 60  # Time steps for LSTM
    output_data = []  # List to store output for CSV

    for csv_file in csv_files:
        stock_name = os.path.basename(csv_file).split('.')[0]  # Get stock name from filename
        print(f"Training model for: {stock_name}")
        scaled_data, scaler = load_data(csv_file)
        
        # Create datasets
        X, y = create_dataset(scaled_data, time_steps)
        
        # Check for sufficient data
        if X.shape[0] == 0 or X.shape[1] != time_steps:
            print(f"Skipping {csv_file} due to insufficient data.")
            continue
        
        # Reshape input to be [samples, time steps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Split into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build and train the model
        model = build_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

        # Extract and store weights, biases, and recurrent weights
        weights_data = extract_weights_biases(model)
        
        # Append the stock name and model parameters to output data
        for layer_idx, layer_data in enumerate(weights_data):
            output_data.append({
                "stock_name": stock_name,
                "layer": layer_idx + 1,
                "W": layer_data["W"],
                "U": layer_data["U"],
                "B": layer_data["B"],
            })

    # Check if the file exists to decide if the header should be written
    file_exists = os.path.isfile('Backend/model/training_output.csv')

    # Save the output data to CSV in append mode
    if output_data:
        output_df = pd.DataFrame(output_data)
        output_df.to_csv('Backend/model/training_output.csv', mode='a', header=not file_exists, index=False)

# Function to parallelize training across multiple CSV files
# Function to parallelize training across multiple CSV files
def parallel_training():
    csv_folder = 'Backend/model/pre_processed_data'  # Specify the path to your CSV files
    csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith('.csv')]

    # Only take the first 10 CSV files
    csv_files = csv_files[:10]

    num_threads = min(4, len(csv_files))  # Limit the number of threads to the number of files

    with multiprocessing.Pool(processes=num_threads) as pool:
        pool.map(train_model_for_stock, [[csv_file] for csv_file in csv_files])  # Each thread gets one file

if __name__ == '__main__':
    parallel_training()
