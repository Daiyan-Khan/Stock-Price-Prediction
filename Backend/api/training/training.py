from flask import Flask, jsonify
from model.lstm import master_task  # Import the necessary function from lstm.py
import os

app = Flask(__name__)

# Define the directory where your data files are located
DATA_DIR = 'pre_processed_data'  # Change this to your actual data directory
TIME_STEPS = 60  # You can adjust this as needed

@app.route('/train', methods=['POST'])
def train():
    """API endpoint to trigger the training of the LSTM model."""
    try:
        # Get all CSV files from the data directory
        csv_files = [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR) if file.endswith('.csv')]

        if not csv_files:
            return jsonify({"message": "No CSV files found for training."}), 404
        
        # Call the master_task function to start training
        master_task(csv_files, TIME_STEPS)

        return jsonify({"message": "Training started for all available stocks."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # You can change the port as needed
