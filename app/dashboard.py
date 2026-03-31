import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

# -----------------------------
# LOAD DATA + MODEL
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/featured_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def load_model():
    model_data = joblib.load("models/model.pkl")
    return model_data["model"], model_data["features"]


df = load_data()
model, features = load_model()

st.title("🦠 Epidemic Outbreak Prediction Dashboard")

# -----------------------------
# TABS (GLOBAL FIRST)
# -----------------------------
tab1, tab2 = st.tabs(["🌍 Global Insights", "📊 Country Analysis"])


# =========================================================
# 🌍 TAB 1: GLOBAL INSIGHTS
# =========================================================
with tab1:

    # Latest snapshot
    latest_global = df.sort_values("Date").groupby("Country/Region").tail(1).copy()

    latest_global["Predicted"] = np.expm1(
        model.predict(latest_global[features])
    )

    # -----------------------------
    # 🔥 ANIMATED HEATMAP (TIMELINE)
    # -----------------------------
    st.subheader("🌍 Outbreak Spread Over Time")

    df_anim = df.copy()
    df_anim["Date_str"] = df_anim["Date"].dt.strftime("%Y-%m-%d")

    fig_anim = px.scatter_geo(
        df_anim,
        lat="Lat",
        lon="Long",
        size="Cases",
        color="Cases",
        hover_name="Country/Region",
        animation_frame="Date_str",
        title="Global Outbreak Evolution",
        projection="natural earth"
    )

    st.plotly_chart(fig_anim)

    # -----------------------------
    # 🔥 BUBBLE CHART (HOTSPOT ANALYSIS)
    # -----------------------------
    st.subheader("🔥 Hotspot Analysis (Growth vs Spread)")

    fig_bubble = px.scatter(
        latest_global,
        x="Growth_Rate",
        y="Rolling_7",
        size="Predicted",
        color="Predicted",
        hover_name="Country/Region",
        title="Growth vs Cases vs Severity",
        labels={
            "Growth_Rate": "How Fast Cases Are Growing",
            "Rolling_7": "Average Daily Cases (Last 7 Days)",
            "Predicted": "Predicted New Cases"
        }
    )

    st.plotly_chart(fig_bubble)

    
    # -----------------------------
    # 🔥 TOP 10 COUNTRIES
    # -----------------------------
    st.subheader("🔥 Top 10 High-Risk Countries")

    top10 = latest_global.sort_values("Predicted", ascending=False).head(10)

    st.dataframe(
        top10[["Country/Region", "Predicted"]]
        .rename(columns={"Predicted": "Predicted Daily Cases"})
    )

    fig_top10 = px.bar(
        top10,
        x="Country/Region",
        y="Predicted",
        title="Top 10 Predicted Outbreak Countries"
    )

    st.plotly_chart(fig_top10)


# =========================================================
# 📊 TAB 2: COUNTRY ANALYSIS
# =========================================================
with tab2:

    countries = sorted(df["Country/Region"].unique())
    selected_country = st.selectbox("🌍 Select Country", countries)

    country_df = df[df["Country/Region"] == selected_country]

    # -----------------------------
    # HISTORICAL CASES
    # -----------------------------
    st.subheader("📈 Historical Cases")

    fig = px.line(
        country_df,
        x="Date",
        y="Cases",
        title=f"Cases Trend - {selected_country}"
    )
    st.plotly_chart(fig)

    # -----------------------------
    # NEXT DAY PREDICTION
    # -----------------------------
    st.subheader("🔮 Next Day Prediction")

    latest_data = country_df.sort_values("Date").iloc[-1]
    input_features = latest_data[features].values.reshape(1, -1)

    pred_log = model.predict(input_features)
    prediction = np.expm1(pred_log)[0]

    st.metric("Predicted Daily Cases (Next Day)", int(prediction))

    # -----------------------------
    # 7-DAY FORECAST
    # -----------------------------
    st.subheader("📅 7-Day Forecast")

    forecast_days = 7

    recent_values = list(
        country_df.sort_values("Date")["Daily_Cases"].tail(7).values
    )

    last_row = country_df.sort_values("Date").iloc[-1]
    mobility_values = last_row[features[4:]].values

    future_predictions = []

    for i in range(forecast_days):

        lag_1 = recent_values[-1]
        lag_7 = recent_values[0]
        rolling_7 = np.mean(recent_values)

        # 🔥 Stable growth
        growth = last_row["Growth_Rate"]

        input_data = [
            lag_1,
            lag_7,
            rolling_7,
            growth,
            *mobility_values
        ]

        pred_log = model.predict([input_data])[0]
        pred = np.expm1(pred_log)

        # Remove noise
        if pred < 1:
            pred = 0

        pred = int(pred)

        future_predictions.append(pred)

        recent_values.pop(0)
        recent_values.append(pred)

    future_dates = pd.date_range(
        start=last_row["Date"] + pd.Timedelta(days=1),
        periods=forecast_days
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Cases": future_predictions
    })

    fig_forecast = px.line(
        forecast_df,
        x="Date",
        y="Predicted Cases",
        title=f"7-Day Forecast - {selected_country}"
    )

    st.plotly_chart(fig_forecast)
    st.dataframe(forecast_df)

    # -----------------------------
    # RISK LEVEL
    # -----------------------------
    st.subheader("🚨 Risk Level")

    if prediction > 10000:
        risk = "HIGH 🔴"
    elif prediction > 1000:
        risk = "MEDIUM 🟠"
    else:
        risk = "LOW 🟢"

    st.write(f"### {risk}")