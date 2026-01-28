import io
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

from model import MLPRegressor

st.set_page_config(
    page_title="Saudi Weather Dashboard",
    page_icon="📊",
    layout="wide",
)

# =========================
# Load model assets
# =========================
@st.cache_resource
def load_model_assets():
    with open("assets/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("assets/ohe_columns.pkl", "rb") as f:
        ohe_cols = pickle.load(f)

    model = MLPRegressor(in_features=len(ohe_cols))
    state = torch.load("weights/mlp_weather_state_dict.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, scaler, ohe_cols

def load_default_data():
    # Optional default dataset stored in repo
    try:
        return pd.read_csv("data/saudi_weather_sample.csv")
    except Exception:
        return None

def preprocess_for_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize datetime column if exists
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # Ensure common numeric columns exist
    # (No strict requirements for dashboard; we only visualize if columns exist)
    return df

def make_features_row(station_name, city, season, month, hour, dew, wind, visibility, dayofweek, day, is_weekend):
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

def preprocess_row_for_model(df_row: pd.DataFrame, ohe_cols: list, scaler):
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
# App header
# =========================
st.title("📊 Saudi Weather Dashboard + Temperature Prediction")
st.caption("Explore the dataset and make a temperature prediction using a PyTorch MLP model.")

# =========================
# Sidebar: data source
# =========================
st.sidebar.header("Data Source")

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
use_default = st.sidebar.checkbox("Use bundled sample dataset (if available)", value=True)

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    if use_default:
        df = load_default_data()

if df is None:
    st.warning("No dataset loaded. Upload a CSV or add data/saudi_weather_sample.csv to the repo.")
    st.stop()

df = preprocess_for_dashboard(df)

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["Dashboard", "Prediction"])

# =========================
# TAB 1: Dashboard
# =========================
with tab1:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")

    if "city" in df.columns:
        c3.metric("Cities", f"{df['city'].nunique():,}")
    else:
        c3.metric("Cities", "—")

    if "station_name" in df.columns:
        c4.metric("Stations", f"{df['station_name'].nunique():,}")
    else:
        c4.metric("Stations", "—")

    with st.expander("Show data table"):
        st.dataframe(df.head(200), use_container_width=True)

    st.divider()
    st.subheader("Filters")

    filt_df = df.copy()

    colA, colB, colC = st.columns(3)

    if "city" in filt_df.columns:
        cities = sorted([c for c in filt_df["city"].dropna().unique().tolist()])
        sel_cities = colA.multiselect("City", cities, default=cities[:1] if len(cities) else [])
        if sel_cities:
            filt_df = filt_df[filt_df["city"].isin(sel_cities)]

    if "station_name" in filt_df.columns:
        stations = sorted([s for s in filt_df["station_name"].dropna().unique().tolist()])
        sel_st = colB.multiselect("Station", stations, default=stations[:1] if len(stations) else [])
        if sel_st:
            filt_df = filt_df[filt_df["station_name"].isin(sel_st)]

    if "Season_name" in filt_df.columns:
        seasons = sorted([s for s in filt_df["Season_name"].dropna().unique().tolist()])
        sel_season = colC.multiselect("Season", seasons, default=seasons[:1] if len(seasons) else [])
        if sel_season:
            filt_df = filt_df[filt_df["Season_name"].isin(sel_season)]

    # Time filter if datetime exists
    if "datetime" in filt_df.columns and pd.api.types.is_datetime64_any_dtype(filt_df["datetime"]):
        min_dt = filt_df["datetime"].min()
        max_dt = filt_df["datetime"].max()
        if pd.notna(min_dt) and pd.notna(max_dt):
            st.caption("Date range")
            start, end = st.slider(
                "Datetime",
                min_value=min_dt.to_pydatetime(),
                max_value=max_dt.to_pydatetime(),
                value=(min_dt.to_pydatetime(), max_dt.to_pydatetime()),
            )
            filt_df = filt_df[(filt_df["datetime"] >= pd.to_datetime(start)) & (filt_df["datetime"] <= pd.to_datetime(end))]

    st.divider()
    st.subheader("Visuals")

    # 1) Temperature distribution
    if "air_temperature" in filt_df.columns:
        st.write("Air temperature distribution")
        fig, ax = plt.subplots()
        vals = pd.to_numeric(filt_df["air_temperature"], errors="coerce").dropna().values
        if len(vals) > 0:
            ax.hist(vals, bins=30)
            ax.set_xlabel("Air temperature (°C)")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("No valid air_temperature values after filtering.")
    else:
        st.info("Column not found: air_temperature")

    # 2) Temperature by hour (mean)
    if "air_temperature" in filt_df.columns and "hour" in filt_df.columns:
        st.write("Average temperature by hour")
        temp = pd.to_numeric(filt_df["air_temperature"], errors="coerce")
        hr = pd.to_numeric(filt_df["hour"], errors="coerce")
        tmp = filt_df.copy()
        tmp["air_temperature"] = temp
        tmp["hour"] = hr
        tmp = tmp.dropna(subset=["air_temperature", "hour"])
        if len(tmp) > 0:
            grp = tmp.groupby("hour")["air_temperature"].mean().sort_index()
            fig, ax = plt.subplots()
            ax.plot(grp.index.values, grp.values)
            ax.set_xlabel("Hour")
            ax.set_ylabel("Avg air temperature (°C)")
            st.pyplot(fig)
        else:
            st.info("Not enough valid data to plot by hour.")
    else:
        st.info("Columns not found: air_temperature and/or hour")

    # 3) Boxplot by season (if exists)
    if "air_temperature" in filt_df.columns and "Season_name" in filt_df.columns:
        st.write("Temperature by season")
        tmp = filt_df.copy()
        tmp["air_temperature"] = pd.to_numeric(tmp["air_temperature"], errors="coerce")
        tmp = tmp.dropna(subset=["air_temperature", "Season_name"])
        if len(tmp) > 0:
            seasons = tmp["Season_name"].unique().tolist()
            data = [tmp[tmp["Season_name"] == s]["air_temperature"].values for s in seasons]
            fig, ax = plt.subplots()
            ax.boxplot(data, labels=seasons, showfliers=False)
            ax.set_ylabel("Air temperature (°C)")
            st.pyplot(fig)
        else:
            st.info("Not enough valid data to plot by season.")
    else:
        st.info("Columns not found: air_temperature and/or Season_name")

# =========================
# TAB 2: Prediction
# =========================
with tab2:
    st.subheader("Temperature Prediction")

    model, scaler, ohe_cols = load_model_assets()

    # Use dataset to provide dropdowns if available; fallback to text inputs
    col1, col2, col3 = st.columns(3)

    if "station_name" in df.columns:
        station_options = sorted(df["station_name"].dropna().unique().tolist())
        station_name = col1.selectbox("Station name", station_options[:200] if len(station_options) else ["ABHA"])
    else:
        station_name = col1.text_input("Station name", value="ABHA")

    if "city" in df.columns:
        city_options = sorted(df["city"].dropna().unique().tolist())
        city = col2.selectbox("City", city_options[:200] if len(city_options) else ["ABHA"])
    else:
        city = col2.text_input("City", value="ABHA")

    if "Season_name" in df.columns:
        season_options = sorted(df["Season_name"].dropna().unique().tolist())
        season = col3.selectbox("Season", season_options if len(season_options) else ["Summer", "Winter", "Spring", "Autumn"])
    else:
        season = col3.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"], index=2)

    colA, colB, colC = st.columns(3)
    month = colA.slider("Month", 1, 12, 8)
    hour = colB.slider("Hour", 0, 23, 21)
    dew = colC.number_input("Dew point (°C)", value=16.0, step=0.5)

    colD, colE = st.columns(2)
    wind = colD.number_input("Wind speed", value=1.2, step=0.1)
    visibility = colE.number_input("Visibility distance", value=10000.0, step=100.0)

    with st.expander("Advanced (optional)"):
        dayofweek = st.slider("Day of week (0=Mon ... 6=Sun)", 0, 6, 0)
        day = st.slider("Day of month", 1, 31, 1)
        is_weekend = st.selectbox("Is weekend?", [0, 1], index=0)

    if st.button("Predict"):
        row = make_features_row(
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
            X = preprocess_row_for_model(row, ohe_cols=ohe_cols, scaler=scaler)
            pred = predict_temp(model, X)
            st.success(f"Predicted air temperature: **{pred:.2f} °C**")
        except Exception:
            st.error("Prediction failed. Ensure the saved one-hot columns and scaler match the training pipeline.")
