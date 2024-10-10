import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import MinMaxScaler

def process_csv(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Example data processing steps
    # 1. Drop rows with any missing values
    data = data.dropna()

    # 2. Normalize specific feature columns (assuming there's a column named 'Close')
    if 'Close' in data.columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data['Close'] = scaler.fit_transform(data[['Close']])

    # 3. Feature extraction (example: adding moving average)
    data['Moving_Average'] = data['Close'].rolling(window=5).mean()  # 5-day moving average

    # 4. Optionally drop any rows with NaN values created by rolling operation
    data = data.dropna()

    # Extract the filename to create the new filename
    file_name = os.path.basename(file_path)
    stock_symbol = file_name.split('_')[0]  # Assuming format is {symbol}_data.csv
    processed_file_name = f"{stock_symbol}_preprocessed_data.csv"
    
    # Define output directory
    output_directory = 'Backend/pre_processed_data'
    os.makedirs(output_directory, exist_ok=True)
    
    # Save the processed data to the output directory
    processed_data_path = os.path.join(output_directory, processed_file_name)
    data.to_csv(processed_data_path, index=False)
    print(f"Processed file saved: {processed_data_path}")

def main():
    # Path to the directory containing the CSV files
    input_directory = 'Backend/model/individual_stocks_5yr/individual_stocks_5yr'
    
    # Get a list of all CSV files in the directory
    csv_files = [os.path.join(input_directory, file) for file in os.listdir(input_directory) if file.endswith('_data.csv')]

    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(process_csv, csv_files)

if __name__ == "__main__":
    main()
