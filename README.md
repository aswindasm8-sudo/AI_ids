 🛡 AI-Powered Intrusion Detection System (IDS)
An AI-driven Intrusion Detection System that detects malicious network activity in real time using an unsupervised deep learning Autoencoder model trained on the CICIDS2017 dataset.
Supports offline training & evaluation and live packet capture for real-world anomaly detection.

🚀 Features
Anomaly-Based Detection – Uses an Autoencoder trained only on benign traffic to detect anomalies.

Offline Mode – Preprocess & analyze CICIDS2017 dataset CSVs.

Live Mode – Sniffs real-time network traffic and detects anomalies instantly.

Flow Reconstruction – Groups packets by (SrcIP, DstIP, SrcPort, DstPort, Protocol) before analysis.

Alerting System – Prints alerts to console & logs them to alerts.log with timestamps & anomaly scores.

Customizable Threshold – Adjust anomaly sensitivity without retraining.

🏗 Tech Stack
Python 3.10+

TensorFlow/Keras – Autoencoder neural network

Scikit-learn – Preprocessing & scaling

Pandas / NumPy – Data handling

Scapy – Live packet capture and flow extraction

Joblib – Save/load pre-trained scaler

📂 Project Structure
text
ai_ids_project/
├── data_preprocessing.py     # Dataset cleaning & preprocessing
├── autoencoder_train.py      # Train AI model on benign traffic
├── real_time_detector.py     # Detect anomalies from CSV streams
├── live_ids_detector.py      # Detect anomalies from live captured packets
├── autoencoder_model.h5      # Trained Autoencoder model
├── scaler.save               # Saved StandardScaler from training
├── alerts.log                # Logs all anomaly alerts
└── data/
    └── cicids2017/           # Dataset files
⚙ Installation
Clone the repository

bash
git clone https://github.com/yourusername/ai-ids-project.git
cd ai-ids-project
Create & activate a virtual environment

bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
Install dependencies

bash
pip install -r requirements.txt
(Windows only) Install Npcap with WinPcap compatibility mode enabled for live packet capture.

📊 Usage
1. Offline Training
bash
python autoencoder_train.py
Loads CICIDS2017 CSV files

Preprocesses & scales features

Trains Autoencoder on benign samples

Saves autoencoder_model.h5 and scaler.save

Prints a suggested anomaly threshold

2. Real-Time Detection from CSV Streams
Place new CSV files into data/stream/ and run:

bash
python real_time_detector.py
3. Live Network Traffic Detection
Run as Administrator/root:

bash
python live_ids_detector.py
Captures packets & extracts flow features

Predicts anomalies using trained Autoencoder

Prints/logs alerts in real time

📡 Example Alert
text
[ALERT] 2025-08-12 18:05:12 | Flow: ('192.168.1.10', '8.8.8.8', 56789, 53, 17) | Anomaly score: 0.034512
This will also be saved in alerts.log for later review.

🧠 How It Works
Data Preprocessing → Clean and scale CICIDS2017 dataset.

Model Training → Autoencoder learns normal network flow patterns.

Threshold Setting → Define cut-off based on reconstruction error.

Detection → Calculate reconstruction error for new flows; alerts triggered if above threshold.
