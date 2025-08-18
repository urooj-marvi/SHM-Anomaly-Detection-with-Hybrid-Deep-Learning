import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow as tf

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="SHM Dashboard", page_icon="📊", layout="wide")
st.markdown("<h1 style='text-align: center;'>🏗️ Structural Health Monitoring (SHM) Dashboard</h1>", unsafe_allow_html=True)

# Sidebar settings
st.sidebar.header("⚙️ Settings")
seq_len = st.sidebar.slider("Sequence Length", min_value=12, max_value=48, value=12, step=1)

# ----------------------------
# File uploader
# ----------------------------
uploaded_file = st.file_uploader("📂 Upload Preprocessed CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["index"])
    df = df.set_index("index").sort_index()
    st.success(f"✅ Data loaded successfully! Shape: {df.shape}")

    # ----------------------------
    # Tabs
    # ----------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Sensor Trends", "⚡ Structural Condition", "🔮 Predictions", "📊 Degradation & Forecast", "📑 Data Table"]
    )

    # ----------------------------
    # Tab 1: Sensor Trends
    # ----------------------------
    with tab1:
        st.subheader("📈 Sensor Time Series Trends")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(df, y=["acceleration_x", "acceleration_y", "acceleration_z"],
                          title="Acceleration (x,y,z)", labels={"value": "Acceleration (g)"})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(df, y=["temperature_c", "humidity_percent", "wind_speed_mps"],
                          title="Environmental Conditions")
            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Tab 2: Structural Condition
    # ----------------------------
    with tab2:
        st.subheader("⚡ Structural Condition Over Time")
        if "damage_class" in df.columns:
            fig = px.scatter(df, x=df.index, y="damage_class", color="damage_class",
                             title="Structural Condition Timeline",
                             labels={"damage_class": "Condition"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No `damage_class` column found in dataset.")

    # ----------------------------
    # Tab 3: Predictions
    # ----------------------------
    with tab3:
        st.subheader("🔮 Model Predictions")
        try:
            model = tf.keras.models.load_model("hybrid_shm_model.keras", compile=False)
            classes = pd.read_csv("label_classes.csv")["class"].tolist()

            # Features
            features = ["acceleration_x","acceleration_y","acceleration_z",
                        "temperature_c","humidity_percent","wind_speed_mps",
                        "fft_peak_freq","fft_magnitude"]
            X = df[features].fillna(method="ffill").values

            # Take first sequence
            X = np.expand_dims(X[:seq_len], axis=0)
            preds = model.predict(X)
            pred_class = classes[np.argmax(preds)]

            st.metric("Predicted Class", pred_class)
            st.progress(float(np.max(preds)))

            fig = px.bar(x=classes, y=preds[0],
                         title="Prediction Confidence",
                         labels={"x": "Class", "y": "Probability"})
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"⚠️ Prediction unavailable: {e}")

    # ----------------------------
    # Tab 4: Degradation & Forecast
    # ----------------------------
    with tab4:
        st.subheader("📊 Degradation & Forecast Trends")
        if "degradation_score" in df.columns and "forecast_score_next_30d" in df.columns:
            fig = px.line(df, y=["degradation_score", "forecast_score_next_30d"],
                          title="Degradation vs Forecast (Next 30 Days)",
                          labels={"value": "Score"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Required columns missing (`degradation_score`, `forecast_score_next_30d`).")

    # ----------------------------
    # Tab 5: Data Table
    # ----------------------------
    with tab5:
        st.subheader("📑 Data Table Preview")
        st.dataframe(df.head(50))

else:
    st.info("📌 Upload a CSV file to start using the dashboard.")
