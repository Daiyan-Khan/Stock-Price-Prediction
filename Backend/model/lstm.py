from mpi4py import MPI
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import io
import warnings

# Ignore TensorFlow specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='tensorflow')

# Other TensorFlow imports and code...

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Set the encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Get the current script directory dynamically
script_dir = os.path.dirname(os.path.abspath(__file__))

# Directories for saving models and data
model_save_dir = os.path.join(script_dir, 'Backend', 'model', 'trained_models')
data_dir = os.path.join(script_dir, 'pre_processed_data')  # Directory where CSV files are stored

# Ensure directories exist
os.makedirs(model_save_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Function to build the LSTM model
def build_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True, stateful=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(128, return_sequences=False, stateful=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)  # Output layer for predicting a single value
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to create dataset for training
def create_training_dataset(data, time_steps):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def load_data(csv_file):
    data = pd.read_csv(csv_file, encoding='utf-8')  # Specify encoding
    prices = data['close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(prices)
    return scaled_data, scaler

# Function to save the trained model
def save_model(model, stock_name):
    save_path = os.path.join(model_save_dir, f"{stock_name}_lstm_model.keras")  # Change .h5 to .keras
    model.save(save_path)
    print(f"Model saved to {save_path}")

# Function to train the LSTM model
def train_model(csv_file, time_steps):
    stock_name = os.path.basename(csv_file).split('.')[0]
    print(f"Training model for stock: {stock_name} on rank {rank}")

    # Load and preprocess data
    scaled_data, _ = load_data(csv_file)

    # Indicate that training has started
    print(f"Started training on {stock_name} after loading the CSV.")

    # Prepare training dataset
    X_train, y_train = create_training_dataset(scaled_data, time_steps)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Build the model
    model = build_lstm_model((X_train.shape[1], 1))

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Save the model
    save_model(model, stock_name)

    # Notify that the model has been trained
    print(f"{stock_name} model has been trained.")

def master_task(csv_files, time_steps):
    num_files = len(csv_files)

    # Handle the case where no slave processes are available
    if size <= 1:
        print("\nOnly the master is running. Processing all files directly. \n")
        for file in csv_files:
            train_model(file, time_steps)
        return

    chunk_size = num_files // (size - 1)

    # Distribute work to slaves
    for i in range(1, size):
        start_idx = (i - 1) * chunk_size
        if i == size - 1:  # Last chunk might have more if num_files is not divisible evenly
            end_idx = num_files
        else:
            end_idx = start_idx + chunk_size

        comm.send(csv_files[start_idx:end_idx], dest=i, tag=i)

# Slave function to receive work and perform computations
def slave_task(time_steps):
    files = comm.recv(source=0, tag=rank)
    print(f"Rank {rank} received {len(files)} files to process.")
    for file in files:
        train_model(file, time_steps)

# Main block
if __name__ == "__main__":
    time_steps = 60

    if rank == 0:
        # Master: Get files and distribute tasks
        csv_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.csv')]
        # Limit to first 10 CSV files for testing
        csv_files = csv_files[:10]  # Get only the first 10 files for testing
        if len(csv_files) == 0:
            print(f"No CSV files found in {data_dir}")
        else:
            master_task(csv_files, time_steps)
    else:
        # Slave: Receive and process assigned files
        slave_task(time_steps)

# Finalize the MPI environment
MPI.Finalize()
