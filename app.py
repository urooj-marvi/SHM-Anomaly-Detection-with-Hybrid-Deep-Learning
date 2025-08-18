import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ​​​ Load Model & Classes
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("hybrid_shm_model.keras", compile=False)
    classes = pd.read_csv("label_classes.csv")["class"].tolist()
    return model, classes

model, CLASSES = load_model_and_classes()

# ​​​ Sidebar Filters
st.set_page_config(page_title="SHM Anomaly Dashboard", layout="wide")
st.title("🏗 SHM Anomaly Detection Dashboard")

uploaded = st.file_uploader("Upload Preprocessed Data (.CSV)", type="csv")
seq_length = st.sidebar.slider("Sequence length (time steps)", 12, 48, 12)
bridge_sel = None
sensor_sel = None
df = None

if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["index"]).set_index("index")
    st.sidebar.write(f"Data loaded: {df.shape[0]} samples, {df.shape[1]} features")

    # Bridge / Sensor selectors
    bridge_sel = st.sidebar.selectbox("Select Bridge", sorted(df["bridge_id"].unique()))
    sensor_sel = st.sidebar.selectbox(
        "Select Sensor", sorted(df[df["bridge_id"] == bridge_sel]["sensor_id"].unique())
    )

# ​​​ Sequence Building
def build_sequences(grouped, vib_cols, env_cols, win):
    vib, env, timestamps = [], [], []
    if len(grouped) < win:
        return np.empty((0, win, len(vib_cols))), np.empty((0, win, len(env_cols))), []
    grp = grouped[vib_cols + env_cols].interpolate(limit_direction="both").values
    for start in range(len(grouped) - win + 1):
        vib.append(grp[start:start+win, :len(vib_cols)])
        env.append(grp[start:start+win, len(vib_cols):])
        timestamps.append(grouped.index[start + win - 1])
    return np.array(vib), np.array(env), timestamps

# ​​​ Prediction + Dashboard
if df is not None and bridge_sel and sensor_sel:
    df_sel = df[(df["bridge_id"] == bridge_sel) & (df["sensor_id"] == sensor_sel)]
    vib_cols = ["acceleration_x","acceleration_y","acceleration_z"]
    env_cols = [c for c in ["temperature_c","wind_speed_mps","humidity_percent",
                            "fft_peak_freq","fft_magnitude","degradation_score","forecast_score_next_30d"] 
                if c in df_sel.columns]

    Xv, Xe, ts = build_sequences(df_sel, vib_cols, env_cols, seq_length)

    if Xv.size > 0:
        preds = model.predict({"vib_in": Xv, "env_in": Xe}, verbose=0)
        pred_idx = np.argmax(preds, axis=1)
        pred_labels = [CLASSES[i] for i in pred_idx]
        confidences = np.max(preds, axis=1)

        results = pd.DataFrame({
            "timestamp": ts,
            "predicted": pred_labels,
            "confidence": confidences
        }).set_index("timestamp")

        # ​​​ Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Time Series", "Download", "About"])

        with tab1:
            st.header(" Anomaly Summary")
            st.write(results["predicted"].value_counts().rename("Count"))
            st.bar_chart(results["predicted"].value_counts())

        with tab2:
            st.header(" Sensor Data & Prediction Timeline")
            fig, ax = plt.subplots(figsize=(10, 6))
            df_plot = df_sel.loc[results.index]
            for col in vib_cols + env_cols:
                ax.plot(df_plot.index, df_plot[col], label=col)
            ax2 = ax.twinx()
            ax2.scatter(results.index, results["predicted"].astype("category").cat.codes,
                        c=results["predicted"].astype("category").cat.codes, cmap="tab10", label="Prediction", s=50)
            ax2.set_yticks(range(len(CLASSES)))
            ax2.set_yticklabels(CLASSES)
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")
            st.pyplot(fig)

        with tab3:
            st.header(" Download Predictions")
            st.write("Save your predictions as CSV for reports or further analysis")
            st.download_button(" Download CSV", results.to_csv(), "predictions.csv")

        with tab4:
            st.header(" About")
            st.markdown("""
This app performs **real-time structural health anomaly detection** using a hybrid deep learning model.
- **Architecture**: CNN + BiLSTM + Transformer fusion
- **Input**: 10-min synchronized sensor data
- **Output**: Anomaly class + confidence

Use the sidebar filters to select the bridge, sensor, and sequence window.
            """)
    else:
        st.error("Not enough data for the selected sequence length.")

