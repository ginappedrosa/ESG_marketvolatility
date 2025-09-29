import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import yfinance as yf
import numpy as np
import os

# ========================
# MAIN PAGE INFO
# ========================
st.title("📊 ESG & Market Volatility Dashboard")
st.markdown("""
Welcome to the **ESG & Market Volatility Dashboard** 🌍📈  

This tool uses a **machine learning model trained on S&P500 companies** to predict stock volatility.  

🔹 **Options available now:**  
- Select **tickers from our ESG dataset** (includes ESG scores + market data).  
- Or enter **any other US ticker** (via Yahoo Finance) to estimate volatility.  

⚠️ **Note:** Predictions are reliable only for US tickers, since the model was trained with S&P500 data.  

🚧 **Next steps (coming soon):**  
- Expand coverage to **IBEX35 (Spain)**, **Eurostoxx50 (Europe)**, and other international indices.  
- Enrich predictions with additional macro & ESG data.  

Use the tabs below to explore companies, compare ESG vs volatility, run predictions, or simulate portfolios.

**Created by:** Gina Pedrosa, Erika Pablos, and Lielia Rodas
""")

# ========================
# PATHS
# ========================
MODEL_PATH = "src/data/lgbm_mix_model.pkl"
METRICS_PATH = "src/data/lgbm_mix_model_metrics.json"
DATASET_PATH = "src/data/dataset_final.csv"
CATEGORICAL_RULES_PATH = "src/data/categorical_rules.json"

# ========================
# STREAMLIT CONFIG
# ========================
st.set_page_config(
    page_title="ESG & Market Volatility Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)
port = int(os.environ.get("PORT", 8501))

# ========================
# STYLING
# ========================
st.markdown("""
<style>
body {background: linear-gradient(to bottom right, #F5F7FA, #C3CFE2); font-family: 'Inter', sans-serif;}
h1, h2, h3 {color: #2E8B57; margin-bottom: 10px;}
.stButton>button {background-color: #2E8B57; color: white;}
.stPlotlyChart {margin-top: 15px; margin-bottom: 25px;}
</style>
""", unsafe_allow_html=True)

# ========================
# LOADERS
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
    df = df.drop(columns=["CEO Full Name", "CEO Status"], errors="ignore")
    return df

@st.cache_data
def load_categorical_rules():
    with open(CATEGORICAL_RULES_PATH, "r") as f:
        return json.load(f)

# ========================
# CATEGORICAL ENCODING
# ========================
def apply_categorical_encoding(df, rules):
    for col, mapping in rules.items():
        if col in df.columns:
            df[f"{col}_n"] = df[col].map(mapping).fillna(-1).astype(int)
        else:
            df[f"{col}_n"] = -1
    return df

# ========================
# INIT
# ========================
model = load_model()
metrics = load_metrics()
data = load_data()
categorical_rules = load_categorical_rules()
data = apply_categorical_encoding(data, categorical_rules)

TRAINING_FEATURES = [
    "Open","High","Low","Close","Adj Close","Volume",
    "ESG Score","Governance Score","Environment Score","Social Score",
    "Year","Daily_Return"
] + [f"{col}_n" for col in categorical_rules.keys()] + ["DUMMY_FILL"]

# ========================
# DEBUG FUNCTION
# ========================
def print_debug(X, context):
    with open("debug_features.txt", "a") as f:
        f.write(f"\n==== {context} ====\n")
        f.write(f"X.shape: {X.shape}\n")
        f.write(f"X.columns: {list(X.columns)}\n")
        f.write(f"TRAINING_FEATURES ({len(TRAINING_FEATURES)}): {TRAINING_FEATURES}\n")
        model_features = getattr(model, 'feature_name_', None)
        f.write(f"model.feature_name_: {model_features}\n")

# ========================
# PREDICTION FUNCTIONS
# ========================
def predict_with_dataset(ticker: str):
    ticker_data = data[data["Ticker"] == ticker].copy()
    X = pd.DataFrame(index=ticker_data.index)
    for col in TRAINING_FEATURES:
        X[col] = ticker_data[col] if col in ticker_data.columns else -1
    X = X[TRAINING_FEATURES]
    print_debug(X, "predict_with_dataset")
    y_pred = model.predict(X)
    return ticker_data, y_pred

def predict_with_yfinance(ticker: str):
    yf_data = yf.download(ticker, period="1y", progress=False)
    if yf_data.empty:
        return None, None
    yf_data["Return"] = yf_data["Adj Close"].pct_change()
    yf_data["Daily_Volatility"] = yf_data["Return"].rolling(window=30).std() * np.sqrt(252)
    X_new = pd.DataFrame(index=yf_data.index)
    for col in TRAINING_FEATURES:
        X_new[col] = yf_data[col].fillna(0) if col in yf_data.columns else -1
    X_new = X_new[TRAINING_FEATURES]
    print_debug(X_new, "predict_with_yfinance")
    y_pred = model.predict(X_new.tail(1))
    return yf_data, y_pred

# ========================
# SIDEBAR FILTERS
# ========================
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

# ========================
# GLOBAL TICKER INPUT
# ========================
st.sidebar.header("Ticker Input")
ticker_input = st.sidebar.text_input("Enter any US ticker symbol:", "AAPL").upper()

if ticker_input in data["Ticker"].unique():
    df_ticker, preds = predict_with_dataset(ticker_input)
    ticker_source = "dataset"
else:
    df_ticker, preds = predict_with_yfinance(ticker_input)
    ticker_source = "yfinance"

if df_ticker is None or df_ticker.empty:
    st.sidebar.error(f"Ticker {ticker_input} not found or cannot fetch data.")
    df_ticker = pd.DataFrame()
    preds = []

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

# TAB 1 - Company Overview
with tab1:
    st.header(f"Company Overview - {ticker_input} ({ticker_source})")
    st.markdown("""
    **Description:** See key metrics of the selected company.  
    **Contains:** Average ESG Score, Daily Volatility, Adjusted Close price.  
    **Usage:** Quickly check ESG performance and market stability. Hover over rows to explore values.
    """)
    if not df_ticker.empty:
        kpi_data = df_ticker[["ESG Score", "Daily_Volatility", "Adj Close"]].mean().to_frame().T
        st.dataframe(kpi_data.style.background_gradient(cmap="Greens", subset=["ESG Score"])
                                     .highlight_max(subset=["Adj Close"], color="lightblue"))

# TAB 2 - ESG vs Volatility
with tab2:
    st.header(f"ESG Score vs Daily Volatility - {ticker_input} ({ticker_source})")
    st.markdown("""
    **Description:** Scatter plot of ESG Score vs Daily Volatility.  
    **Contains:** Each point represents a company, sized by stock price.  
    **Usage:** Identify if higher ESG scores are associated with lower volatility.
    """)
    if ticker_source == "dataset" and not df_ticker.empty:
        fig2 = px.scatter(
            df_ticker, x="ESG Score", y="Daily_Volatility",
            hover_data=["Ticker","Adj Close","Governance Score","Environment Score","Social Score"],
            size="Adj Close", color_discrete_sequence=["#2E8B57"]
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("ESG data not available for this ticker.")

# TAB 3 - Prediction
with tab3:
    st.header(f"Predict Volatility - {ticker_input} ({ticker_source})")
    st.markdown("""
    **Description:** Forecast short-term daily volatility for a company.  
    **Usage:** Enter a ticker, view historical or estimated volatility. Predicted volatility is shown for the latest date.
    """)
    if not df_ticker.empty:
        fig_pred = go.Figure()
        x_vals = df_ticker.index if ticker_source=="yfinance" else df_ticker["Date"]
        fig_pred.add_trace(go.Scatter(x=x_vals, y=df_ticker["Daily_Volatility"], mode="lines",
                                      name="Volatility", line=dict(color="#1F77B4", width=3)))
        st.plotly_chart(fig_pred, use_container_width=True)
        if len(preds)>0:
            st.metric("Predicted Volatility (latest)", f"{preds[-1]:.4f}")

# TAB 4 - Portfolio Simulation
with tab4:
    st.header("Portfolio Simulation")
    st.markdown("""
    **Description:** Analyze selected tickers as a portfolio.  
    **Contains:** Evolution of daily volatility, average volatility per ticker.  
    **Usage:** Select tickers in sidebar. Assess combined portfolio risk and ESG trade-offs.
    """)
    if selected_tickers:
        port_data = data[data["Ticker"].isin(selected_tickers)]
        fig_port = px.line(port_data, x="Date", y="Daily_Volatility", color="Ticker",
                           title="Portfolio Volatility Evolution",
                           color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_port, use_container_width=True)
        avg_vol = port_data.groupby("Ticker")["Daily_Volatility"].mean()
        st.bar_chart(avg_vol)

# TAB 5 - Model Performance
with tab5:
    st.header("Model Performance")
    st.markdown("""
    **Description:** View predictive model metrics and feature importance.  
    **Contains:** R², MSE, MAE, and importance of features.  
    **Usage:** Understand model quality and main drivers of volatility predictions.
    """)
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{metrics.get('R2_test',0):.3f}")
    col2.metric("MSE", f"{metrics.get('MSE_test',0):.3f}")
    col3.metric("MAE", f"{metrics.get('MAE_test',0):.3f}")
    if "feature_importance" in metrics:
        fig_imp = px.bar(
            x=metrics["feature_importance"]["features"],
            y=metrics["feature_importance"]["importance"],
            title="Feature Importance",
            color=metrics["feature_importance"]["importance"],
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
