import os
import requests
import pandas as pd
import schedule
import time
import api.data_service.data_service as data_service
import api.fetch_new_data.fetch_new_data as fetch_new_data
import api.training.training as training
import api.prediction.prediction as prediction

def job():
    """The scheduled job to fetch data, preprocess, and train the model."""
    fetch_new_data.update_all_stocks
    data_service.preprocess_and_store
    training.train
    prediction.predict
# Schedule the job every day at a specific time (e.g., 09:00 AM)
schedule.every().day.at("09:00").do(job)

# Run the scheduler
while True:
    schedule.run_pending()
    time.sleep(1)  # Wait for 1 second before checking the schedule again
