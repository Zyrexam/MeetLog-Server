# ambient_client.py

import requests
import datetime
import json

# Ambient server base URL
AMBIENT_SERVER_URL = "http://192.168.0.123/"

# File to store ambient logs
LOG_FILE = "ambient_logs.jsonl"

def fetch_ambient_data():
    """
    Fetch ambient data from ambient server root URL.
    Returns dict or None if failed.
    """
    try:
        response = requests.get(AMBIENT_SERVER_URL, timeout=3)
        if response.status_code == 200:
            data = response.json()
            print("Fetched ambient data:", data)
            return data
        else:
            print(f"Failed to fetch ambient data: status {response.status_code}")
    except Exception as e:
        print(f"Error connecting to ambient server: {e}")
    return None

def save_ambient_data(data):
    """
    Save ambient data with timestamp to local log file.
    """
    if not data:
        return
    data_with_time = {
        "fetched_at": datetime.datetime.now().isoformat(),
        **data
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data_with_time) + "\n")
    print("Saved ambient data to log.")

def fetch_and_log_ambient_data():
    """
    Fetch from server and save to log.
    """
    data = fetch_ambient_data()
    save_ambient_data(data)

if __name__ == "__main__":
    # Test: python3 ambient_client.py
    fetch_and_log_ambient_data()
