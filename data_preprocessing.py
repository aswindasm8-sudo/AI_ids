import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "data/cicids2017"

def load_data(file_list):
    dfs = []
    for file in file_list:
        path = os.path.join(DATA_DIR, file)
        print(f"Loading {path}...")
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()    # <-- strip spaces from column names
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    return data


def preprocess_data(df):
    # Drop irrelevant columns if any (already done)
    df = df.drop(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], axis=1, errors='ignore')

    # Strip column whitespace if not done already
    df.columns = df.columns.str.strip()

    # Handle missing and infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)  # or use df.dropna() to remove rows with NaNs

    # Convert labels to binary: Attack or Benign
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    # Encode categorical features - for example Protocol
    if 'Protocol' in df.columns:
        le = LabelEncoder()
        df['Protocol'] = le.fit_transform(df['Protocol'])

    # Separate features and label
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def plot_class_distribution(y):
    sns.countplot(y)
    plt.title("Class Distribution")
    plt.show()

if __name__ == "__main__":
    # Example usage
    files = ['Monday-WorkingHours.pcap_ISCX.csv', 'Tuesday-WorkingHours.pcap_ISCX.csv']
    df = load_data(files)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    plot_class_distribution(df['Label'])
    X, y = preprocess_data(df)
    print(f"Preprocessed features shape: {X.shape}")
