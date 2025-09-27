import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import yfinance as yf
import numpy as np
# ========================
# PATHS
# ========================
MODEL_PATH = "src/data/lgbm_mix_model.pkl"
METRICS_PATH = "src/data/lgbm_mix_model_metrics.json"
DATASET_PATH = "src/data/dataset_final.csv"
# ========================
# STREAMLIT APP CONFIG
# ========================
st.set_page_config(
    page_title="ESG & Market Volatility Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)
# ========================
# STYLING - PROFESSIONAL LOOK
# ========================
st.markdown("""
<style>
body {
    background: linear-gradient(to bottom right, #F5F7FA, #C3CFE2);
    font-family: 'Helvetica', 'Arial', sans-serif;
}
h1, h2, h3, h4, h5 {
    color: #2E8B57;
}
.stButton>button {
    background-color: #2E8B57;
    color: white;
}
</style>
""", unsafe_allow_html=True)
# ========================
# LOAD DATA & MODEL
# ========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r") as f:
        return json.load(f)
@st.cache_data
def load_data():
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=["CEO Full Name", "CEO Status"], errors='ignore')
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    return df, numeric_cols
model = load_model()
metrics = load_metrics()
data, numeric_cols = load_data()
# ========================
# PREDICTION FUNCTIONS
# ========================
def predict_with_dataset(ticker: str):
    ticker_data = data[data["Ticker"] == ticker].copy()
    X = ticker_data[numeric_cols].drop(columns=["Daily_Volatility"], errors='ignore')
    y_pred = model.predict(X)
    return ticker_data, y_pred
def predict_with_yfinance(ticker: str):
    yf_data = yf.download(ticker, period="1y", progress=False)
    if yf_data.empty:
        return None, None
    yf_data["Return"] = yf_data["Adj Close"].pct_change()
    yf_data["Volatility"] = yf_data["Return"].rolling(window=30).std() * np.sqrt(252)
    X_new = yf_data[["Open", "High", "Low", "Close", "Volume"]].fillna(0).tail(1)
    y_pred = model.predict(X_new)
    return yf_data, y_pred
# ========================
# SIDEBAR FILTERS
# ========================
st.sidebar.header("Filters & Company Selection")
esg_min, esg_max = st.sidebar.slider("ESG Score Range:", float(data["ESG Score"].min()), float(data["ESG Score"].max()), (0.0, 100.0))
vol_min, vol_max = st.sidebar.slider("Volatility Range:", float(data["Daily_Volatility"].min()), float(data["Daily_Volatility"].max()), (0.0, 0.1))
year_select = st.sidebar.multiselect("Year:", options=data["Year"].unique(), default=data["Year"].unique())
selected_tickers = st.sidebar.multiselect("Select Tickers:", options=data["Ticker"].unique(), default=data["Ticker"].unique()[:5])
filtered_data = data[
    (data["ESG Score"] >= esg_min) & (data["ESG Score"] <= esg_max) &
    (data["Daily_Volatility"] >= vol_min) & (data["Daily_Volatility"] <= vol_max) &
    (data["Year"].isin(year_select)) &
    (data["Ticker"].isin(selected_tickers))
]
# ========================
# TABS
# ========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    ":office: Company Overview",
    ":earth_africa: ESG vs Volatility",
    ":crystal_ball: Prediction",
    ":briefcase: Portfolio Simulation",
    ":cog: Model Performance"
])
# ========================
# TAB 1 - COMPANY OVERVIEW
# ========================
with tab1:
    st.header("Company Ranking & KPIs")
    kpi_data = filtered_data.groupby("Ticker")[["ESG Score", "Daily_Volatility", "Adj Close"]].mean().sort_values(by=["ESG Score", "Daily_Volatility"], ascending=[False, True])
    st.dataframe(kpi_data.style.background_gradient(cmap="Greens", subset=["ESG Score"]).highlight_max(subset=["Adj Close"], color="lightblue"))
# ========================
# TAB 2 - ESG vs VOLATILITY
# ========================
with tab2:
    st.header("ESG Score vs Daily Volatility")
    fig2 = px.scatter(
        filtered_data, x="ESG Score", y="Daily_Volatility",
        color="Ticker", hover_data=["Ticker"],
        color_continuous_scale="Viridis", size="Adj Close"
    )
    st.plotly_chart(fig2, use_container_width=True)
# ========================
# TAB 3 - PREDICTION
# ========================
with tab3:
    st.header("Predict Volatility by Ticker")
    ticker_input = st.text_input("Enter ticker symbol:", "AAPL").upper()
    if ticker_input in data["Ticker"].unique():
        st.success(f"Data for {ticker_input} retrieved from ESG dataset.")
        df, preds = predict_with_dataset(ticker_input)
        fig_pred = px.line(df, x="Date", y="Daily_Volatility", title=f"Historical Volatility - {ticker_input}", color_discrete_sequence=["#2E8B57"])
        st.plotly_chart(fig_pred, use_container_width=True)
        st.metric("Predicted Volatility (latest)", f"{preds[-1]:.4f}")
    else:
        st.warning(f"{ticker_input} not in ESG dataset. Using Yahoo Finance data.")
        df, preds = predict_with_yfinance(ticker_input)
        if df is not None:
            fig_pred = px.line(df, x=df.index, y="Volatility", title=f"Estimated Volatility - {ticker_input}", color_discrete_sequence=["#3CB371"])
            st.plotly_chart(fig_pred, use_container_width=True)
            st.metric("Predicted Volatility", f"{preds[0]:.4f}")
        else:
            st.error("Unable to retrieve data for this ticker.")
# ========================
# TAB 4 - PORTFOLIO SIMULATION
# ========================
with tab4:
    st.header("Portfolio Simulation")
    if selected_tickers:
        port_data = filtered_data
        fig_port = px.line(port_data, x="Date", y="Daily_Volatility", color="Ticker", title="Portfolio Volatility Evolution", color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_port, use_container_width=True)
        avg_vol = port_data.groupby("Ticker")["Daily_Volatility"].mean()
        st.bar_chart(avg_vol)
# ========================
# TAB 5 - MODEL PERFORMANCE
# ========================
with tab5:
    st.header("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{metrics.get('R2_test',0):.3f}")
    col2.metric("MSE", f"{metrics.get('MSE_test',0):.3f}")
    col3.metric("MAE", f"{metrics.get('MAE_test',0):.3f}")
    if "feature_importance" in metrics:
        fig_imp = px.bar(x=metrics["feature_importance"]["features"], y=metrics["feature_importance"]["importance"], title="Feature Importance", color=metrics["feature_importance"]["importance"], color_continuous_scale="Viridis")
        st.plotly_chart(fig_imp, use_container_width=True)


# ========================
# SIDEBAR FILTERS
# ========================
st.sidebar.header("Filters & Company Selection")
esg_min, esg_max = st.sidebar.slider("ESG Score Range:", float(data["ESG Score"].min()), float(data["ESG Score"].max()), (0.0, 100.0))
vol_min, vol_max = st.sidebar.slider("Volatility Range:", float(data["Daily_Volatility"].min()), float(data["Daily_Volatility"].max()), (0.0, 0.1))
year_select = st.sidebar.multiselect("Year:", options=data["Year"].unique(), default=data["Year"].unique())
selected_tickers = st.sidebar.multiselect("Select Tickers:", options=data["Ticker"].unique(), default=data["Ticker"].unique()[:5])
filtered_data = data[
    (data["ESG Score"] >= esg_min) & (data["ESG Score"] <= esg_max) &
    (data["Daily_Volatility"] >= vol_min) & (data["Daily_Volatility"] <= vol_max) &
    (data["Year"].isin(year_select)) &
    (data["Ticker"].isin(selected_tickers))
]
# ========================
# TABS
# ========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    ":office: Company Overview",
    ":earth_africa: ESG vs Volatility",
    ":crystal_ball: Prediction",
    ":briefcase: Portfolio Simulation",
    ":cog: Model Performance"
])
# ========================
# TAB 1 - COMPANY OVERVIEW
# ========================
with tab1:
    st.header("Company Ranking & KPIs")
    kpi_data = filtered_data.groupby("Ticker")[["ESG Score", "Daily_Volatility", "Adj Close"]].mean().sort_values(by=["ESG Score", "Daily_Volatility"], ascending=[False, True])
    st.dataframe(kpi_data.style.background_gradient(cmap="Greens", subset=["ESG Score"]).highlight_max(subset=["Adj Close"], color="lightblue"))
# ========================
# TAB 2 - ESG vs VOLATILITY
# ========================
with tab2:
    st.header("ESG Score vs Daily Volatility")
    fig2 = px.scatter(
        filtered_data, x="ESG Score", y="Daily_Volatility",
        color="Ticker", hover_data=["Ticker"],
        color_continuous_scale="Viridis", size="Adj Close"
    )
    st.plotly_chart(fig2, use_container_width=True)
# ========================
# TAB 3 - PREDICTION
# ========================
with tab3:
    st.header("Predict Volatility by Ticker")
    ticker_input = st.text_input("Enter ticker symbol:", "AAPL").upper()
    if ticker_input in data["Ticker"].unique():
        st.success(f"Data for {ticker_input} retrieved from ESG dataset.")
        df, preds = predict_with_dataset(ticker_input)
        fig_pred = px.line(df, x="Date", y="Daily_Volatility", title=f"Historical Volatility - {ticker_input}", color_discrete_sequence=["#2E8B57"])
        st.plotly_chart(fig_pred, use_container_width=True)
        st.metric("Predicted Volatility (latest)", f"{preds[-1]:.4f}")
    else:
        st.warning(f"{ticker_input} not in ESG dataset. Using Yahoo Finance data.")
        df, preds = predict_with_yfinance(ticker_input)
        if df is not None:
            fig_pred = px.line(df, x=df.index, y="Volatility", title=f"Estimated Volatility - {ticker_input}", color_discrete_sequence=["#3CB371"])
            st.plotly_chart(fig_pred, use_container_width=True)
            st.metric("Predicted Volatility", f"{preds[0]:.4f}")
        else:
            st.error("Unable to retrieve data for this ticker.")
# ========================
# TAB 4 - PORTFOLIO SIMULATION
# ========================
with tab4:
    st.header("Portfolio Simulation")
    if selected_tickers:
        port_data = filtered_data
        fig_port = px.line(port_data, x="Date", y="Daily_Volatility", color="Ticker", title="Portfolio Volatility Evolution", color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_port, use_container_width=True)
        avg_vol = port_data.groupby("Ticker")["Daily_Volatility"].mean()
        st.bar_chart(avg_vol)
# ========================
# TAB 5 - MODEL PERFORMANCE
# ========================
with tab5:
    st.header("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{metrics.get('R2_test',0):.3f}")
    col2.metric("MSE", f"{metrics.get('MSE_test',0):.3f}")
    col3.metric("MAE", f"{metrics.get('MAE_test',0):.3f}")
    if "feature_importance" in metrics:
        fig_imp = px.bar(x=metrics["feature_importance"]["features"], y=metrics["feature_importance"]["importance"], title="Feature Importance", color=metrics["feature_importance"]["importance"], color_continuous_scale="Viridis")
        st.plotly_chart(fig_imp, use_container_width=True)