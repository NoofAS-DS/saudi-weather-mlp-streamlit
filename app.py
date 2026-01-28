import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle, json
import matplotlib.pyplot as plt
from model import MLPRegressor

st.set_page_config("Saudi Weather Forecast", layout="wide")

# ================= Load Assets =================
@st.cache_resource
def load_assets():
    with open("assets/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("assets/ohe_columns.pkl", "rb") as f:
        ohe_cols = pickle.load(f)

    with open("assets/metrics.json") as f:
        metrics = json.load(f)

    loss_df = pd.read_csv("assets/loss_curve.csv")
    stats = pd.read_csv("assets/sample_stats.csv", index_col=0)

    model = MLPRegressor(len(ohe_cols))
    model.load_state_dict(torch.load("weights/mlp_weather_state_dict.pt", map_location="cpu"))
    model.eval()

    return model, scaler, ohe_cols, metrics, loss_df, stats

model, scaler, ohe_cols, metrics, loss_df, stats = load_assets()

# ================= UI =================
st.title("🌡️ Saudi Weather Temperature Forecast (MLP)")
tabs = st.tabs(["Overview", "Statistics", "Model Performance", "Prediction"])

# -------- Overview --------
with tabs[0]:
    st.subheader("Dataset Overview")
    st.dataframe(stats)

# -------- Statistics --------
with tabs[1]:
    st.subheader("Temperature Distribution")
    fig, ax = plt.subplots()
    ax.hist(stats.loc["mean":"max"].values.flatten())
    st.pyplot(fig)

# -------- Model Performance --------
with tabs[2]:
    st.metric("MAE", metrics["MAE"])
    st.metric("RMSE", metrics["RMSE"])
    st.metric("R²", metrics["R2"])

    fig, ax = plt.subplots()
    ax.plot(loss_df["train_loss"], label="Train")
    ax.plot(loss_df["val_loss"], label="Val")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    st.pyplot(fig)

# -------- Prediction --------
with tabs[3]:
    st.subheader("Predict Temperature")

    station = st.text_input("Station Name", "ABHA")
    city = st.text_input("City", "ABHA")
    season = st.selectbox("Season", ["Winter","Spring","Summer","Autumn"])
    month = st.slider("Month", 1, 12, 8)
    hour = st.slider("Hour", 0, 23, 21)
    wind = st.number_input("Wind Speed", 0.0, 20.0, 1.2)
    visibility = st.number_input("Visibility", 0.0, 20000.0, 10000.0)
    dew = st.number_input("Dew Point", -5.0, 40.0, 16.0)

    if st.button("Predict"):
        row = pd.DataFrame([{
            "station_name": station,
            "city": city,
            "Season_name": season,
            "month": month,
            "hour": hour,
            "wind_speed_rate": wind,
            "visibility_distance": visibility,
            "air_temperature_dew_point": dew,
            "dayofweek": 0,
            "day": 1,
            "is_weekend": 0
        }])

        row_ohe = pd.get_dummies(row)
        for col in ohe_cols:
            if col not in row_ohe.columns:
                row_ohe[col] = 0
        row_ohe = row_ohe[ohe_cols]

        X = scaler.transform(row_ohe).astype(np.float32)
        with torch.no_grad():
            pred = model(torch.tensor(X)).item()

        st.success(f"🌡️ Predicted Temperature: {pred:.2f} °C")
