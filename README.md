## **Multi-Modal Structural Health Monitoring (SHM) – Anomaly Detection**
📖 Overview

This project implements a hybrid deep learning model for anomaly detection in Structural Health Monitoring (SHM).
The system fuses vibration data (accelerometers), environmental conditions (temperature, wind, humidity), and frequency-domain features (FFT) to detect different types of structural anomalies such as Normal, Drift, Minor, and Major.
---
📂** Project Structure**
│── app.py                  # Streamlit dashboard
│── hybrid_shm_model.keras  # Trained hybrid model
│── label_classes.csv       # Class label mapping
│── requirements.txt        # Dependencies
│── sample_data.csv         # Demo dataset
│── README.md               # Project documentation
---
📊 **Data Format**

The input dataset should be a synchronized & preprocessed CSV with the following columns:

index (timestamp)

bridge_id, sensor_id

acceleration_x, acceleration_y, acceleration_z

temperature_c, wind_speed_mps, humidity_percent

fft_peak_freq, fft_magnitude

damage_class (label for training/evaluation)

---
🧠 **Model Architecture**

The anomaly detection model is a hybrid neural network:

CNN (1D) → extracts vibration patterns

BiLSTM → learns temporal dependencies

Transformer Encoder → attention mechanism to focus on critical timesteps

Fusion Layer → combines vibration + environmental features

Dense Layers → final classification into anomaly classes
---
📈 **Model Evaluation**

Accuracy: ~88%

Macro F1-score: ~0.87

K-Fold CV Accuracy: 88.4% ± 0.65%

📌 Observations:

Normal class detected with high recall (~95%)

Most confusion occurs between Minor and Drift anomalies

Multi-modal fusion improves accuracy by ~12% compared to vibration-only models
---
**Dashboard Features**

The Streamlit dashboard provides an interactive anomaly detection interface:

📂 Upload preprocessed sensor data (CSV)

📊 Preview input dataset

⚙️ Adjust sequence length for analysis (e.g., 12 steps = 2 hours)

🤖 Run model predictions → anomaly class + confidence score

📈 Visualize anomalies over time with plots

📥 Download predictions as CSV

---

Deliverables

Preprocessed dataset → Synchronized & Preprocessed.csv

Feature dataset → bridge_features.csv

Trained model → hybrid_shm_model.keras

Label mapping → label_classes.csv

Interactive dashboard → app.py (Streamlit)
---
Author: Urooj Marvi
📧 Contact: uroojmarvi456@gmail.com
