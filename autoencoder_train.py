import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Define your dataset directory
DATA_DIR = "data/cicids2017"

def load_data(file_list):
    dfs = []
    for file in file_list:
        path = os.path.join(DATA_DIR, file)
        print(f"Loading {path}...")
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()  # Strip spaces in column headers
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    return data

def preprocess_data(df):
    # Drop irrelevant or identifying columns
    df = df.drop(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], axis=1, errors='ignore')

    # Clean data: replace infinite with NaN and then fill with 0
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Convert labels to binary: 0 = BENIGN, 1 = Attack
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    # Encode categorical features if exist - example: Protocol
    if 'Protocol' in df.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df['Protocol'] = le.fit_transform(df['Protocol'])

    # Separate features and label
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Scale features with StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

if __name__ == "__main__":
    # List the CSV files you have in your dataset folder
    files = ['Monday-WorkingHours.pcap_ISCX.csv', 'Tuesday-WorkingHours.pcap_ISCX.csv']

    print("[INFO] Loading and preprocessing data...")
    df = load_data(files)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    X_scaled, y, scaler = preprocess_data(df)

    # Save the scaler for consistent preprocessing later in live detection
    joblib.dump(scaler, "scaler.save")
    print("[INFO] Saved scaler to 'scaler.save'")

    # Separate normal and attack samples
    X_normal = X_scaled[y == 0]
    X_attack = X_scaled[y == 1]

    print(f"[INFO] Normal samples: {X_normal.shape[0]}, Attack samples: {X_attack.shape[0]}")

    # Split normal data into training and validation
    X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)

    # Build the autoencoder model
    input_dim = X_train.shape[1]
    encoding_dim = 32

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(64, activation="relu")(input_layer)
    encoder = Dense(encoding_dim, activation="relu")(encoder)
    decoder = Dense(64, activation="relu")(encoder)
    decoder = Dense(input_dim, activation=None)(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)

    # Compile with explicit MeanSquaredError loss for safe saving/loading
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())

    autoencoder.summary()

    # Train the autoencoder only on normal data
    EPOCHS = 20
    BATCH_SIZE = 512

    history = autoencoder.fit(
        X_train, X_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_data=(X_val, X_val)
    )

    # Save the trained model for inference
    autoencoder.save("autoencoder_model.h5")
    print("[INFO] Autoencoder model saved to 'autoencoder_model.h5'")

    # Plot loss curves
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.title("Autoencoder Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Evaluate anomalies on attack data to pick threshold
    reconstructions = autoencoder.predict(X_attack)
    mse = np.mean(np.power(X_attack - reconstructions, 2), axis=1)

    threshold = np.percentile(mse, 95)
    print(f"[INFO] Suggested anomaly threshold (95th percentile of attack MSE): {threshold:.6f}")

    # Test on combined normal and attack data
    all_X = np.vstack((X_normal, X_attack))
    all_y = np.hstack((np.zeros(len(X_normal)), np.ones(len(X_attack))))

    all_recons = autoencoder.predict(all_X)
    errors = np.mean(np.power(all_X - all_recons, 2), axis=1)

    preds = (errors > threshold).astype(int)

    print("\n[Classification Report]")
    print(classification_report(all_y, preds))

    print("\n[Confusion Matrix]")
    print(confusion_matrix(all_y, preds))

    print("\n[ROC-AUC Score]")
    print(roc_auc_score(all_y, errors))
