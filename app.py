import streamlit as st
import pandas as pd
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Mechanical Failure Predictor",
                   layout="centered")

st.title("Mechanical Failure Predictor")
st.markdown("""
Upload a CSV or enter sensor readings to predict failure probability.
""")

uploaded = st.file_uploader(
    "Upload CSV with columns: sensor_1, sensor_2, sensor_3, operating_temp",
    type=["csv"]
)

if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())

        expected = {"sensor_1", "sensor_2", "sensor_3", "operating_temp"}
        if not expected.issubset(df.columns):
            st.error(f"CSV must contain: {sorted(list(expected))}")
        else:
            payload = df.to_dict(orient="records")
            resp = requests.post(f"{BACKEND_URL}/predict_batch", json=payload)
            if resp.status_code == 200:
                preds = resp.json()
                df["failure_probability"] = preds
                st.dataframe(df)
            else:
                st.error(f"Error: {resp.status_code}")
    except Exception as e:
        st.error(f"Error: {e}")

st.divider()
st.header("Single Prediction")

s1 = st.number_input("sensor_1", value=0.12, format="%.3f")
s2 = st.number_input("sensor_2", value=10.0)
s3 = st.number_input("sensor_3", value=100.0)
temp = st.number_input("operating_temp", value=35.0)

if st.button("Predict"):
    try:
        payload = {
            "sensor_1": s1,
            "sensor_2": s2,
            "sensor_3": s3,
            "operating_temp": temp
        }
        resp = requests.post(f"{BACKEND_URL}/predict", json=payload)
        if resp.status_code == 200:
            prob = resp.json().get("failure_probability")
            st.success(f"Failure probability: {prob:.3f}")
        else:
            st.error(f"Error: {resp.status_code}")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("_Backend must be running on localhost:8000_")
