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

This dashboard allows you to explore **US stocks** with ESG scores and predict **daily volatility** using a machine learning model trained on S&P500 companies.

🔹 **Important:**  
- ESG data comes from 2022-2023.  
- Volatility is calculated using the most recent price data from Yahoo Finance, ensuring predictions reflect current market conditions.  
- Predictions are **experimental** and for demonstration purposes, but show potential for understanding ESG vs risk.  

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

def predict_with_current_prices(ticker: str):
    yf_data = yf.download(ticker, period="1y", progress=False)
    if yf_data.empty:
        return None, None
    # Ensure columns exist
    if 'Adj Close' not in yf_data.columns:
        # st.warning(f("'Adj Close' not found for {ticker}, using 'Close' instead."))
        yf_data['Adj Close'] = yf_data['Close']
    yf_data["Return"] = yf_data["Adj Close"].pct_change()
    yf_data["Daily_Volatility"] = yf_data["Return"].rolling(window=30).std() * np.sqrt(252)
    X_new = pd.DataFrame(index=yf_data.index)
    for col in TRAINING_FEATURES:
        X_new[col] = yf_data[col].fillna(0) if col in yf_data.columns else -1
    X_new = X_new[TRAINING_FEATURES]
    print_debug(X_new, "predict_with_current_prices")
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

# Always get latest prices, even for dataset tickers
df_ticker, preds = predict_with_current_prices(ticker_input)
if df_ticker is None or df_ticker.empty:
    st.sidebar.error(f"Ticker {ticker_input} not found or cannot fetch data.")
    df_ticker = pd.DataFrame()
    preds = []

ticker_source = "yfinance"  # Always showing current data

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

# TAB 1
with tab1:
    st.header(f"Company Overview - {ticker_input}")
    st.markdown("""
    **Description:** Key company metrics using latest stock prices.  
    **Contains:** ESG Score, Daily Volatility, Adjusted Close price.  
    **Usage:** Quickly check ESG and market stability.
    """)
    if not df_ticker.empty:
        kpi_data = pd.DataFrame({
            "ESG Score": [data[data["Ticker"]==ticker_input]["ESG Score"].mean()],
            "Daily Volatility": [df_ticker["Daily_Volatility"].mean()],
            "Adj Close": [df_ticker["Adj Close"].mean()]
        })
        st.dataframe(kpi_data.style.background_gradient(cmap="Greens", subset=["ESG Score"])
                                     .highlight_max(subset=["Adj Close"], color="lightblue"))

# TAB 2
with tab2:
    st.header(f"ESG Score vs Daily Volatility - {ticker_input}")
    st.markdown("""
    **Description:** Scatter plot showing the relationship between ESG score and daily volatility.  
    **Note:** ESG score is annual (2022-2023) but volatility uses latest Yahoo Finance data.  
    **Usage:** Understand potential correlation between ESG performance and stock risk.
    """)
    if not df_ticker.empty:
        esg_val = data[data["Ticker"]==ticker_input]["ESG Score"].mean()
        x_vals = np.repeat(esg_val, len(df_ticker))
        y_vals = df_ticker['Daily_Volatility'].values
        fig2 = px.scatter(x=x_vals, y=y_vals, labels={'x':'ESG Score','y':'Daily Volatility'})
        st.plotly_chart(fig2, use_container_width=True)

# TAB 3
with tab3:
    st.header(f"Predict Volatility - {ticker_input}")
    st.markdown("""
    **Description:** Forecast short-term daily volatility using latest market data.  
    **Usage:** Line chart shows historical volatility, metric shows predicted latest value.
    """)
    if not df_ticker.empty:
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=df_ticker.index, y=df_ticker["Daily_Volatility"],
            mode="lines", name="Volatility", line=dict(color="#1F77B4", width=3)
        ))
        st.plotly_chart(fig_pred, use_container_width=True)
        if preds is not None and len(preds) > 0:
            st.metric("Predicted Volatility (latest)", f"{preds[-1]:.4f}")

# TAB 4
with tab4:
    st.header("Portfolio Simulation")
    st.markdown("""
    **Description:** Compare volatility evolution for selected tickers.  
    **Usage:** Assess portfolio risk and ESG trade-offs.
    """)
    if selected_tickers:
        port_data = pd.DataFrame()
        for t in selected_tickers:
            df_t, _ = predict_with_current_prices(t)
            if df_t is not None:
                df_t["Ticker"] = t
                port_data = pd.concat([port_data, df_t])
    if not port_data.empty:
                port_data = port_data.copy()
                # Si las columnas son MultiIndex, conviértelas a columnas simples
                if isinstance(port_data.columns, pd.MultiIndex):
                    port_data.columns = ['_'.join([str(i) for i in col if i]) for col in port_data.columns.values]
                # Si 'Daily_Volatility' no está en las columnas, intenta buscarla
                if 'Daily_Volatility' not in port_data.columns:
                    # Buscar cualquier columna que contenga 'Daily_Volatility'
                    dv_cols = [col for col in port_data.columns if 'Daily_Volatility' in str(col)]
                    if dv_cols:
                        port_data['Daily_Volatility'] = port_data[dv_cols[0]]
                fig_port = px.line(
                    port_data, x=port_data.index, y="Daily_Volatility", color="Ticker",
                    title="Portfolio Volatility Evolution", color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig_port, use_container_width=True)
                avg_vol = port_data.groupby("Ticker")["Daily_Volatility"].mean()
                st.bar_chart(avg_vol)

# TAB 5
with tab5:
    st.header("Model Performance")
    st.markdown("""
    **Description:** View ML model metrics and feature importance.  
    **Contains:** R², MSE, MAE, and importance of features.  
    **Usage:** Understand model quality and main drivers of predictions.
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
