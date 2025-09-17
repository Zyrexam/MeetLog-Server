import paramiko
import time
import os
import pandas as pd

# SSH credentials
SERVER_IP = "10.6.0.56"
PORT = 22
USERNAME = "garvit"
PASSWORD = "garvit"

REMOTE_FILE_PATH = "/home/garvit/sai/sensor_data.csv"
LOCAL_FILE_PATH = "/home/ananya/attentiveness-server/sensor_data_log.csv"

# Define the correct CSV header
CSV_HEADER = "Timestamp,Temperature,Humidity,Light Intensity,Co2 Concentration,Door Status,Motion Status\n"

while True:
    try:
        print("🔄 Fetching remote CSV...")

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(SERVER_IP, PORT, USERNAME, PASSWORD)

        sftp = ssh.open_sftp()
        sftp.get(REMOTE_FILE_PATH, LOCAL_FILE_PATH)
        sftp.close()
        ssh.close()

        print(f"✅ Fetched and saved to {LOCAL_FILE_PATH}")

        # 🔍 Check if header exists
        with open(LOCAL_FILE_PATH, 'r+') as f:
            first_line = f.readline()
            if not first_line.strip().startswith("timestamp"):
                print("⚠️ Header missing, adding header...")
                data = f.read()
                f.seek(0)
                f.write(CSV_HEADER + data)
                f.truncate()
            else:
                print("✅ Header exists.")

        # 📊 Load with pandas & print for debug
        df = pd.read_csv(LOCAL_FILE_PATH)
        print("📄 CSV columns:", df.columns.tolist())
        print(df.tail(3))

    except Exception as e:
        print("❌ Error fetching or processing file:", e)

    time.sleep(5)
