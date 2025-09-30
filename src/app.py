import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import yfinance as yf
import numpy as np
import os


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

# PATHS
MODEL_PATH = "src/data/best_model_XGBoost_mix.pkl"
METRICS_PATH = "src/data/metrics_XGBoost_mix.json"
DATASET_PATH = "src/data/dataset_final.csv"
CATEGORICAL_RULES_PATH = "src/data/categorical_rules.json"

# STREAMLIT CONFIG
st.set_page_config(
    page_title="ESG & Market Volatility Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)
port = int(os.environ.get("PORT", 8501))

# STYLING
st.markdown("""
<style>
body {background: linear-gradient(to bottom right, #F5F7FA, #C3CFE2); font-family: 'Inter', sans-serif;}
h1, h2, h3 {color: #2E8B57; margin-bottom: 10px;}
.stButton>button {background-color: #2E8B57; color: white;}
.stPlotlyChart {margin-top: 15px; margin-bottom: 25px;}
</style>
""", unsafe_allow_html=True)

# LOADERS
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

# CATEGORICAL ENCODING
def apply_categorical_encoding(df, rules):
    for col, mapping in rules.items():
        if col in df.columns:
            df[f"{col}_n"] = df[col].map(mapping).fillna(-1).astype(int)
        else:
            df[f"{col}_n"] = -1
    return df

# INIT
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
    for col in TRAINING_FEATURES:
        X[col] = ticker_data[col] if col in ticker_data.columns else -1
    X = X[TRAINING_FEATURES]
    print_debug(X, "predict_with_dataset")
    y_pred = model.predict(X)
    return ticker_data, y_pred
def predict_with_hybrid(ticker: str):
    yf_data = yf.download(ticker, period="1y", progress=False)
    if yf_data.empty:
        return None, None
    if 'Adj Close' not in yf_data.columns:
        yf_data['Adj Close'] = yf_data['Close']
    yf_data["Return"] = yf_data["Adj Close"].pct_change()
    yf_data["Daily_Volatility"] = yf_data["Return"].rolling(window=30).std() * np.sqrt(252)
    esg_row = data[data["Ticker"] == ticker].iloc[-1] if ticker in data["Ticker"].unique() else None
    X_new = pd.DataFrame(index=yf_data.index)
    for col in TRAINING_FEATURES:
        if col in yf_data.columns:
            X_new[col] = yf_data[col].fillna(0)
        elif esg_row is not None and col in esg_row.index:
            X_new[col] = esg_row[col]
        else:
            X_new[col] = -1
    X_new = X_new[TRAINING_FEATURES]
    print_debug(X_new, "predict_with_hybrid")

    # --- Ajuste de columnas para XGBoost ---
    import pickle
    feature_names_path = "src/data/feature_names_XGBoost_mix.pkl"
    if os.path.exists(feature_names_path):
        with open(feature_names_path, "rb") as f:
            feature_names = pickle.load(f)
        # Añade columnas faltantes y ordena
        for col in feature_names:
            if col not in X_new.columns:
                X_new[col] = -1
        X_new = X_new[feature_names]

    # --- Fin ajuste columnas ---
    y_pred = model.predict(X_new.tail(1))
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
    st.markdown("""
    **Description:** Key company metrics using the latest stock prices.
    **Includes:** ESG Score, Daily Volatility, Adjusted Close price.
    **Usage:** Quickly check ESG and market stability for the selected company.
    """)
    if not df_ticker.empty:
        kpi_data = pd.DataFrame({
            "ESG Score": [data[data["Ticker"]==ticker_input]["ESG Score"].mean()],
            "Daily Volatility": [df_ticker["Daily_Volatility"].mean()],
            "Adj Close": [df_ticker["Adj Close"].mean()]
        })
        st.dataframe(kpi_data.style.background_gradient(cmap="Greens", subset=["ESG Score"])
                                     .highlight_max(subset=["Adj Close"], color="lightblue"))

# TAB 2 - ESG Score vs Daily Volatility
with tab2:
    st.header(f"ESG Score vs Average Volatility - {ticker_input}")
    st.markdown("""
    **Description:** Here you can visually see the relationship between the ESG score and the company's average volatility.
    - **ESG Score (Sustainalytics):** lower is better (less ESG risk, more responsible).
    - **Average volatility:** lower means the stock price is more stable.

    **Easy interpretation:**
    - Point at bottom left: very responsible and stable company (ESG < 20 and low volatility).
    - Point at top right: company with higher ESG risk and more unstable price.
    - Green color means low ESG (<20), yellow medium (20-40), red high (>40).
    """)
    if not df_ticker.empty:
        esg_val = data[data["Ticker"]==ticker_input]["ESG Score"].mean()
        vol_mean = df_ticker["Daily_Volatility"].mean()
        # Color by ESG
        if esg_val < 20:
            color = "green"
        elif esg_val < 40:
            color = "yellow"
        else:
            color = "red"
        fig2 = px.scatter(x=[esg_val], y=[vol_mean], labels={'x':'ESG Score','y':'Average Volatility'}, color_discrete_sequence=[color])
        fig2.update_traces(marker=dict(size=30))
        fig2.update_layout(
            xaxis=dict(range=[0, max(50, esg_val+10)]),
            yaxis=dict(range=[0, max(0.1, vol_mean+0.02)]),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(f"**Company:** {ticker_input} | **ESG Score:** {esg_val:.2f} | **Average Volatility:** {vol_mean:.4f}")
        if esg_val < 20 and vol_mean < 0.02:
            st.success("This company is very responsible and stable!")
        elif esg_val > 40 and vol_mean > 0.05:
            st.error("Company with high ESG risk and very unstable price.")
        else:
            st.info("Company with intermediate sustainability and stability profile.")

# TAB 3- Predict Volatility
with tab3:
    st.header(f"Predict Volatility - {ticker_input}")
    st.markdown("""
    **Description:** Forecast short-term daily volatility using the latest market data.
    **Usage:** The line chart shows historical volatility, and the metric displays the predicted latest value.
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

# TAB 4 - Portfolio Simulation
with tab4:
    st.header("Portfolio Simulation")
    st.markdown("""
    **Description:** Compare the volatility evolution for selected tickers.
    **Usage:** Assess portfolio risk and ESG trade-offs visually.
    """)
    if selected_tickers:
        port_data = pd.DataFrame()
        for t in selected_tickers:
            df_t, _ = predict_with_hybrid(t)
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

# TAB 5 - Model Performance
with tab5:
    st.header("Model Performance")
    st.markdown("""
    **Description:** View machine learning model metrics and feature importance.
    **Includes:** R², RMSE, MAE, and feature importance.
    **Usage:** Understand model quality and the main drivers of predictions.
    """
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{metrics.get('r2_test', 0):.3f}")
    col2.metric("RMSE", f"{metrics.get('rmse_test', 0):.3f}")
    col3.metric("MAE", f"{metrics.get('mae_test', 0):.3f}")

    if "feature_importance" in metrics:
        fig_imp = px.bar(
            x=metrics["feature_importance"]["features"],
            y=metrics["feature_importance"]["importance"],
            title="Feature Importance",
            color=metrics["feature_importance"]["importance"],
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_imp, use_container_width=True)