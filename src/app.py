import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
import yfinance as yf
import numpy as np
import os
import xgboost as xgb

# STREAMLIT CONFIG
st.set_page_config(
    page_title="ESG & Market Volatility Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

# MAIN PAGE INFO
st.title(":bar_chart: ESG & Market Volatility Dashboard")
st.markdown("""
Welcome to the **ESG & Market Volatility Dashboard** :earth_africa::chart_with_upwards_trend:  
This dashboard allows you to explore **US stocks** with ESG scores and predict **daily volatility** using a machine learning model trained on S&P500 companies.  

:small_blue_diamond: **Important:**  
- ESG data comes from 2022-2023.  
- Volatility is calculated using the most recent price data from Yahoo Finance, ensuring predictions reflect current market conditions.  
- Predictions are **experimental** and for demonstration purposes, but show potential for understanding ESG vs risk.  

**Created by:** Gina Pedrosa, Erika Pablos, and Lielia Rodas
""")


# PATHS (siempre relativos a la ubicación de este archivo)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "data/best_model_XGBoost_mix.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "data/feature_names_XGBoost_mix.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "data/metrics_XGBoost_mix.json")
DATASET_PATH = os.path.join(BASE_DIR, "data/dataset_final.csv")
CATEGORICAL_RULES_PATH = os.path.join(BASE_DIR, "data/categorical_rules.json")

# LOADERS
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r") as f:
        return json.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=["CEO Full Name", "CEO Status"], errors="ignore")
    return df

@st.cache_data
def load_categorical_rules():
    with open(CATEGORICAL_RULES_PATH, "r") as f:
        return json.load(f)

# INIT
model = load_model()
metrics = load_metrics()
data = load_data()
categorical_rules = load_categorical_rules()

with open(FEATURES_PATH, "rb") as f:
    feature_names = pickle.load(f)

# CATEGORICAL ENCODING
def apply_categorical_encoding(df, rules):
    for col, mapping in rules.items():
        if col in df.columns:
            df[f"{col}_n"] = df[col].map(mapping).fillna(-1).astype(int)
        else:
            df[f"{col}_n"] = -1
    return df

data = apply_categorical_encoding(data, categorical_rules)

# DEBUG FUNCTION
def print_debug(X, context):
    with open("debug_features.txt", "a") as f:
        f.write(f"\n==== {context} ====\n")
        f.write(f"X.shape: {X.shape}\n")
        f.write(f"X.columns: {list(X.columns)}\n")

# PREDICTION FUNCTIONS
def predict_with_dataset(ticker: str):
    ticker_data = data[data["Ticker"] == ticker].copy()
    X = pd.DataFrame(index=ticker_data.index)
    for col in feature_names:
        X[col] = ticker_data[col] if col in ticker_data.columns else -1
    X = X[feature_names]
    print_debug(X, "predict_with_dataset")

    # Convert to DMatrix
    dtest = xgb.DMatrix(X.to_numpy(dtype=np.float32), feature_names=feature_names)
    y_pred = model.get_booster().predict(dtest)
    return ticker_data, y_pred

def predict_with_hybrid(ticker: str):
    yf_data = yf.download(ticker, period="1y", progress=False)
    if yf_data.empty:
        return None, None

    if "Adj Close" not in yf_data.columns:
        yf_data["Adj Close"] = yf_data["Close"]

    yf_data["Return"] = yf_data["Adj Close"].pct_change()
    yf_data["Daily_Volatility"] = yf_data["Return"].rolling(window=30).std() * np.sqrt(252)

    esg_row = data[data["Ticker"] == ticker].iloc[-1] if ticker in data["Ticker"].unique() else None

    X_new = pd.DataFrame(index=yf_data.index)
    for col in feature_names:
        if col in yf_data.columns:
            X_new[col] = yf_data[col].fillna(0)
        elif esg_row is not None and col in esg_row.index:
            val = esg_row[col]
            # Forzar a escalar puro
            if isinstance(val, (pd.Series, np.ndarray, list)):
                val = val[0]
            X_new[col] = pd.Series([val]*len(X_new), index=X_new.index)
        else:
            X_new[col] = -1

    X_new = X_new[feature_names]
    print_debug(X_new, "predict_with_hybrid")

    # Convert to DMatrix
    dtest = xgb.DMatrix(X_new.tail(1).to_numpy(dtype=np.float32), feature_names=feature_names)
    y_pred = model.get_booster().predict(dtest)
    return yf_data, y_pred

# SIDEBAR FILTERS
st.sidebar.header("Filters & Company Selection")
esg_min, esg_max = st.sidebar.slider("ESG Score Range:",
                                     float(data["ESG Score"].min()),
                                     float(data["ESG Score"].max()),
                                     (0.0, 100.0))
vol_min, vol_max = st.sidebar.slider("Volatility Range:",
                                     float(data["Daily_Volatility"].min()),
                                     float(data["Daily_Volatility"].max()),
                                     (0.0, 0.1))
year_select = st.sidebar.multiselect("Year:",
                                     options=data["Year"].unique(),
                                     default=data["Year"].unique())
selected_tickers = st.sidebar.multiselect("Select Tickers:",
                                          options=data["Ticker"].unique(),
                                          default=data["Ticker"].unique()[:5])

# GLOBAL TICKER INPUT
st.sidebar.header("Ticker Input")
ticker_input = st.sidebar.text_input("Enter any US ticker symbol:", "AAPL").upper()
df_ticker, preds = predict_with_hybrid(ticker_input)
if df_ticker is None or df_ticker.empty:
    st.sidebar.error(f"Ticker {ticker_input} not found or cannot fetch data.")
    df_ticker = pd.DataFrame()
    preds = []

# TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    ":office: Company Overview",
    ":earth_africa: ESG vs Volatility",
    ":crystal_ball: Prediction",
    ":briefcase: Portfolio Simulation",
    ":gear: Model Performance"
])

# TAB 1 - Company Overview
with tab1:
    st.header(f"Company Overview - {ticker_input}")
    if not df_ticker.empty:
        kpi_data = pd.DataFrame({
            "ESG Score": [data[data["Ticker"] == ticker_input]["ESG Score"].mean()],
            "Daily Volatility": [df_ticker["Daily_Volatility"].mean()],
            "Adj Close": [df_ticker["Adj Close"].mean()]
        })
        st.dataframe(kpi_data.style.background_gradient(cmap="Greens", subset=["ESG Score"])
                                     .highlight_max(subset=["Adj Close"], color="lightblue"))

# TAB 2 - ESG Score vs Volatility
with tab2:
    st.header(f"ESG Score vs Average Volatility - {ticker_input}")
    if not df_ticker.empty:
        esg_val = data[data["Ticker"] == ticker_input]["ESG Score"].mean()
        vol_mean = df_ticker["Daily_Volatility"].mean()
        if esg_val < 20:
            color = "green"
        elif esg_val < 40:
            color = "yellow"
        else:
            color = "red"
        fig2 = px.scatter(x=[esg_val], y=[vol_mean], labels={'x': 'ESG Score', 'y': 'Average Volatility'}, color_discrete_sequence=[color])
        fig2.update_traces(marker=dict(size=30))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"**Company:** {ticker_input} | **ESG Score:** {esg_val:.2f} | **Average Volatility:** {vol_mean:.4f}")

# TAB 3 - Predict Volatility
with tab3:
    st.header(f"Predict Volatility - {ticker_input}")
    if not df_ticker.empty:
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=df_ticker.index, y=df_ticker["Daily_Volatility"],
            mode="lines", name="Volatility", line=dict(color="#1F77B4", width=3)
        ))
        st.plotly_chart(fig_pred, use_container_width=True)
        if preds is not None and len(preds) > 0:
            st.metric("Predicted Volatility (latest)", f"{preds[-1]:.4f}")

# TAB 4 - Portfolio Simulation
with tab4:
    st.header("Portfolio Simulation")
    if selected_tickers:
        port_data = pd.DataFrame()
        for t in selected_tickers:
            df_t, _ = predict_with_hybrid(t)
            if df_t is not None:
                df_t["Ticker"] = t
                port_data = pd.concat([port_data, df_t])
        if not port_data.empty:
            if isinstance(port_data.columns, pd.MultiIndex):
                port_data.columns = ['_'.join([str(i) for i in col if i]) for col in port_data.columns]
            fig_port = px.line(
                port_data, x=port_data.index, y="Daily_Volatility", color="Ticker",
                title="Portfolio Volatility Evolution", color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_port, use_container_width=True)

# TAB 5 - Model Performance
with tab5:
    st.header("Model Performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{metrics.get('r2_test', 0):.3f}")
    col2.metric("RMSE", f"{metrics.get('rmse_test', 0):.3f}")
    col3.metric("MAE", f"{metrics.get('mae_test', 0):.3f}")







