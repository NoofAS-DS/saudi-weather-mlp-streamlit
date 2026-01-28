import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

from model import MLPRegressor

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Saudi Weather — Temperature Predictor",
    page_icon="🌡️",
    layout="centered",
)

# =========================
# Safe loaders (optional files)
# =========================
def safe_load_json(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None

def safe_load_csv(path: str):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def safe_load_stats(path: str):
    try:
        return pd.read_csv(path, index_col=0)
    except Exception:
        return None

@st.cache_resource
def load_assets():
    # Required assets
    with open("assets/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("assets/ohe_columns.pkl", "rb") as f:
        ohe_cols = pickle.load(f)

    model = MLPRegressor(in_features=len(ohe_cols))
    state = torch.load("weights/mlp_weather_state_dict.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # Optional assets
    metrics = safe_load_json("assets/metrics.json")
    loss_df = safe_load_csv("assets/loss_curve.csv")
    stats = safe_load_stats("assets/sample_stats.csv")

    return model, scaler, ohe_cols, metrics, loss_df, stats

def make_row(station_name, city, season, month, hour, dew, wind, visibility, dayofweek, day, is_weekend):
    return pd.DataFrame([{
        "station_name": station_name,
        "city": city,
        "Season_name": season,
        "month": int(month),
        "hour": int(hour),
        "air_temperature_dew_point": float(dew),
        "wind_speed_rate": float(wind),
        "visibility_distance": float(visibility),
        "dayofweek": int(dayofweek),
        "day": int(day),
        "is_weekend": int(is_weekend),
    }])

def preprocess_row(df_row: pd.DataFrame, ohe_cols: list, scaler):
    row_ohe = pd.get_dummies(
        df_row,
        columns=["station_name", "city", "Season_name"],
        drop_first=False
    )

    for col in ohe_cols:
        if col not in row_ohe.columns:
            row_ohe[col] = 0

    row_ohe = row_ohe[ohe_cols]
    X = scaler.transform(row_ohe).astype(np.float32)
    return X

def predict_temp(model, X_np: np.ndarray) -> float:
    x_t = torch.tensor(X_np, dtype=torch.float32)
    with torch.no_grad():
        pred = model(x_t).cpu().numpy().ravel()[0]
    return float(pred)

# =========================
# Load model assets
# =========================
try:
    model, scaler, ohe_cols, metrics, loss_df, stats = load_assets()
except Exception:
    st.error("Failed to load model assets. Please verify assets/ and weights/ files are present and correctly named.")
    st.stop()

# =========================
# Header
# =========================
st.title("🌡️ Saudi Weather — Temperature Predictor")
st.caption("A simple PyTorch MLP regression app for predicting air temperature (demo/portfolio).")

# Optional: quick KPIs
if metrics:
    c1, c2, c3 = st.columns(3)
    mae = metrics.get("MAE", None)
    rmse = metrics.get("RMSE", None)
    r2 = metrics.get("R2", None)
    if mae is not None:  c1.metric("MAE", f"{mae:.2f}")
    if rmse is not None: c2.metric("RMSE", f"{rmse:.2f}")
    if r2 is not None:   c3.metric("R²", f"{r2:.3f}")

st.divider()

# =========================
# Main: inputs (simple)
# =========================
st.subheader("Inputs")
with st.form("predict_form", clear_on_submit=False):
    station_name = st.text_input("Station name", value="ABHA")
    city = st.text_input("City", value="ABHA")

    season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"], index=2)

    colA, colB = st.columns(2)
    with colA:
        month = st.slider("Month", 1, 12, 8)
        hour = st.slider("Hour", 0, 23, 21)
    with colB:
        dew = st.number_input("Dew point (°C)", value=16.0, step=0.5)
        wind = st.number_input("Wind speed", value=1.2, step=0.1)

    visibility = st.number_input("Visibility distance", value=10000.0, step=100.0)

    with st.expander("Advanced (optional)"):
        dayofweek = st.slider("Day of week (0=Mon ... 6=Sun)", 0, 6, 0)
        day = st.slider("Day of month", 1, 31, 1)
        is_weekend = st.selectbox("Is weekend?", [0, 1], index=0)

    submit = st.form_submit_button("Predict")

# =========================
# Prediction output
# =========================
if submit:
    row = make_row(
        station_name=station_name,
        city=city,
        season=season,
        month=month,
        hour=hour,
        dew=dew,
        wind=wind,
        visibility=visibility,
        dayofweek=dayofweek,
        day=day,
        is_weekend=is_weekend,
    )

    try:
        X = preprocess_row(row, ohe_cols=ohe_cols, scaler=scaler)
        pred = predict_temp(model, X)
        st.success(f"Predicted air temperature: **{pred:.2f} °C**")
    except Exception:
        st.error("Prediction failed. This is usually caused by a mismatch between saved one-hot columns/scaler and the app inputs.")
        st.stop()

    st.divider()

# =========================
# Lightweight insights
# =========================
st.subheader("Quick Insights")
col1, col2 = st.columns(2)

with col1:
    if stats is not None:
        st.write("Summary statistics")
        st.dataframe(stats, use_container_width=True)
    else:
        st.info("Optional file not found: assets/sample_stats.csv")

with col2:
    if loss_df is not None and {"train_loss", "val_loss"}.issubset(loss_df.columns):
        st.write("Training curve (MSE)")
        fig, ax = plt.subplots()
        ax.plot(loss_df["train_loss"].values, label="Train")
        ax.plot(loss_df["val_loss"].values, label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Optional file not found: assets/loss_curve.csv")

st.caption("Note: This app is for demo purposes only and should not be used for critical decisions.")

