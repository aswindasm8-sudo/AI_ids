import numpy as np
import pandas as pd
import os
import time
from tensorflow.keras.models import load_model
import joblib
from data_preprocessing import preprocess_data, load_data, DATA_DIR
from sklearn.preprocessing import StandardScaler

# -------------------------
# Config
# -------------------------
MODEL_PATH = "autoencoder_model.h5"
THRESHOLD = 0.02   # <-- set from Step 2 output (adjust after training)
STREAM_DATA_DIR = "data/stream"  # folder where new CSVs appear

# -------------------------
# Load model
# -------------------------
print("[INFO] Loading model...")
autoencoder = load_model(MODEL_PATH)

# -------------------------
# Function to detect anomalies
# -------------------------
def detect_anomalies(X_scaled):
    recons = autoencoder.predict(X_scaled)
    errors = np.mean(np.power(X_scaled - recons, 2), axis=1)
    preds = (errors > THRESHOLD).astype(int)
    return preds, errors

# -------------------------
# Watch folder for new files
# -------------------------
print(f"[INFO] Watching folder: {os.path.abspath(STREAM_DATA_DIR)}")
done_files = set()

while True:
    # find new CSV files
    csv_files = [f for f in os.listdir(STREAM_DATA_DIR) if f.endswith(".csv")]
    new_files = [f for f in csv_files if f not in done_files]

    for file in new_files:
        print(f"[INFO] Processing {file}...")
        full_path = os.path.join(STREAM_DATA_DIR, file)

        # Load and preprocess
        df = pd.read_csv(full_path)
        df.columns = df.columns.str.strip()  # clean col names

        if 'Label' in df.columns:
            X_scaled, y = preprocess_data(df)
        else:
            X = df.fillna(0)
            if 'Label' in X.columns:
                X = X.drop('Label', axis=1)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            y = None

        # Detect anomalies
        preds, errors = detect_anomalies(X_scaled)

        anomalies = np.sum(preds)
        total = len(preds)

        print(f"[RESULT] {anomalies}/{total} anomalies detected in {file}")
        if y is not None:
            from sklearn.metrics import classification_report
            print(classification_report(y, preds))

        # mark as processed
        done_files.add(file)

    time.sleep(5)  # check every 5 seconds
