from scapy.all import sniff, IP, TCP, UDP
import time
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import load_model
import joblib

# ----------------------------
# Simple Alert Printer + Logger
# ----------------------------
def alert(flow_key, error):
    """
    Simple alert printer and logger
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    message = f"[ALERT] {timestamp} | Flow: {flow_key} | Anomaly score: {error:.6f}"

    # Print to console
    print(message)

    # Save to log file
    with open("alerts.log", "a") as f:
        f.write(message + "\n")


# ----------------------------
# Load Model and Scaler
# ----------------------------
autoencoder = load_model("autoencoder_model.h5")
scaler = joblib.load("scaler.save")  # Must be saved during training in autoencoder_train.py

THRESHOLD = 0.02  # Adjust based on training evaluation

# Flow tracking dict
flows = defaultdict(lambda: {"packets": [], "timestamps": [], "flags": []})

# Flow timeout in seconds
FLOW_TIMEOUT = 5.0


# ----------------------------
# Feature Extraction
# ----------------------------
def extract_features(flow_packets, flow_times, flow_flags):
    """
    Extract basic CICIDS2017-like features from packets in one flow.
    """
    if len(flow_packets) == 0:
        return None

    durations = flow_times[-1] - flow_times[0]
    packet_lengths = np.array([len(pkt) for pkt in flow_packets])

    # Basic stats
    total_packets = len(packet_lengths)
    total_bytes = np.sum(packet_lengths)
    min_len = np.min(packet_lengths)
    max_len = np.max(packet_lengths)
    mean_len = np.mean(packet_lengths)
    std_len = np.std(packet_lengths)

    # TCP flag counts
    flag_counts = {"SYN": 0, "ACK": 0, "FIN": 0, "RST": 0, "PSH": 0, "URG": 0}
    for flags in flow_flags:
        if flags is None:
            continue
        if flags & 0x02:  # SYN
            flag_counts["SYN"] += 1
        if flags & 0x10:  # ACK
            flag_counts["ACK"] += 1
        if flags & 0x01:  # FIN
            flag_counts["FIN"] += 1
        if flags & 0x04:  # RST
            flag_counts["RST"] += 1
        if flags & 0x08:  # PSH
            flag_counts["PSH"] += 1
        if flags & 0x20:  # URG
            flag_counts["URG"] += 1

    # Feature vector (keep order consistent with training!!)
    feature_vector = [
        durations,
        total_packets,
        total_bytes,
        min_len,
        max_len,
        mean_len,
        std_len,
        flag_counts["SYN"],
        flag_counts["ACK"],
        flag_counts["FIN"],
        flag_counts["RST"],
        flag_counts["PSH"],
        flag_counts["URG"],
    ]

    return np.array(feature_vector)


# ----------------------------
# Expire and Process Flows
# ----------------------------
def check_flows_expiry():
    """
    Check for expired flows based on inactivity.
    Expired flows will be processed into feature vectors.
    """
    current_time = time.time()
    expired = []
    feature_vectors = []
    keys = list(flows.keys())

    for flow_key in keys:
        flow = flows[flow_key]
        if flow["timestamps"] and (current_time - flow["timestamps"][-1] > FLOW_TIMEOUT):
            feat = extract_features(flow["packets"], flow["timestamps"], flow["flags"])
            if feat is not None:
                feature_vectors.append((flow_key, feat))
            expired.append(flow_key)

    # Remove expired flows
    for key in expired:
        del flows[key]

    return feature_vectors


# ----------------------------
# Packet Handler
# ----------------------------
def packet_callback(pkt):
    """
    Callback function for each captured packet.
    Groups packets into flows based on 5-tuple.
    """
    if IP not in pkt:
        return

    ip_layer = pkt[IP]
    proto = ip_layer.proto
    src_ip = ip_layer.src
    dst_ip = ip_layer.dst
    src_port = None
    dst_port = None
    flags = None

    if proto == 6 and TCP in pkt:  # TCP
        tcp_layer = pkt[TCP]
        src_port = tcp_layer.sport
        dst_port = tcp_layer.dport
        flags = tcp_layer.flags
    elif proto == 17 and UDP in pkt:  # UDP
        udp_layer = pkt[UDP]
        src_port = udp_layer.sport
        dst_port = udp_layer.dport

    if src_port is None or dst_port is None:
        return

    flow_key = (src_ip, dst_ip, src_port, dst_port, proto)

    flows[flow_key]["packets"].append(pkt)
    flows[flow_key]["timestamps"].append(time.time())
    flows[flow_key]["flags"].append(flags)


# ----------------------------
# Main Function
# ----------------------------
def main():
    print("Starting live network packet capture...")
    try:
        while True:
            sniff(timeout=1, prn=packet_callback, store=False)

            # Process expired flows
            feature_vectors = check_flows_expiry()

            if feature_vectors:
                feats = np.array([fv[1] for fv in feature_vectors])
                feats_scaled = scaler.transform(feats)

                reconstructions = autoencoder.predict(feats_scaled)
                errors = np.mean(np.power(feats_scaled - reconstructions, 2), axis=1)

                for i, (flow_key, _) in enumerate(feature_vectors):
                    if errors[i] > THRESHOLD:
                        alert(flow_key, errors[i])

    except KeyboardInterrupt:
        print("Stopped by user.")


if __name__ == "__main__":
    main()
