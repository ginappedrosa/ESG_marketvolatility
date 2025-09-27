import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import json
import yfinance as yf
import numpy as np

# ========================
# PATHS - Define where all your data, model, and rules are stored
# ========================
MODEL_PATH = "src/data/lgbm_mix_model.pkl"  # Trained LightGBM model
METRICS_PATH = "src/data/lgbm_mix_model_metrics.json"  # Model performance metrics
DATASET_PATH = "src/data/dataset_final.csv"  # ESG + market dataset
CATEGORICAL_RULES_PATH = "src/data/categorical_rules.json"  # Mapping for categorical columns

# ========================
# STREAMLIT APP CONFIG
# ========================
st.set_page_config(
    page_title="ESG & Market Volatility Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

# For Render deployment
import os
port = int(os.environ.get("PORT", 8501))

# ========================
# STYLING
# ========================
st.markdown("""
<style>
body {background: linear-gradient(to bottom right, #F5F7FA, #C3CFE2); font-family: 'Helvetica', 'Arial', sans-serif;}
h1, h2, h3, h4, h5 {color: #2E8B57;}
.stButton>button {background-color: #2E8B57; color: white;}
</style>
""", unsafe_allow_html=True)

# ========================
# LOADERS - Load model, metrics, dataset, and categorical mappings
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
# CATEGORICAL ENCODING - Convert categorical columns to numeric for model
# ========================
def apply_categorical_encoding(df, rules):
    for col, mapping in rules.items():
        if col in df.columns:
            df[f"{col}_n"] = df[col].map(mapping).fillna(-1).astype(int)
        else:
            df[f"{col}_n"] = -1
    return df

# ========================
# INIT - Load all resources
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
] + [f"{col}_n" for col in categorical_rules.keys()] + ["DUMMY_FILL"]  # Extra dummy column to match model

# ========================
# DEBUG FUNCTION
# ========================
def print_debug(X, context):
    """Print features used for prediction to a debug file for troubleshooting."""
    with open("debug_features.txt", "a") as f:
        f.write(f"\n==== {context} ====\n")
        f.write(f"X.shape: {X.shape}\n")
        f.write(f"X.columns: {list(X.columns)}\n")
        f.write(f"TRAINING_FEATURES ({len(TRAINING_FEATURES)}): {TRAINING_FEATURES}\n")
        model_features = getattr(model, 'feature_name_', None)
        f.write(f"model.feature_name_: {model_features}\n")
        if model_features and len(TRAINING_FEATURES) == len(model_features):
            f.write("\nMAP TRAINING_FEATURES -> model.feature_name_:\n")
            for i, (tf, mf) in enumerate(zip(TRAINING_FEATURES, model_features)):
                f.write(f"{i}: {tf} -> {mf}\n")
        elif model_features:
            f.write(f"\n[ALERT] Column count mismatch: {len(TRAINING_FEATURES)} vs {len(model_features)}\n")
            for i, mf in enumerate(model_features):
                tf = TRAINING_FEATURES[i] if i < len(TRAINING_FEATURES) else '---'
                f.write(f"{i}: {tf} -> {mf}\n")

# ========================
# PREDICTION FUNCTIONS
# ========================
def predict_with_dataset(ticker: str):
    """Predict volatility using internal ESG dataset."""
    ticker_data = data[data["Ticker"] == ticker].copy()
    X = pd.DataFrame(index=ticker_data.index)
    for col in TRAINING_FEATURES:
        X[col] = ticker_data[col] if col in ticker_data.columns else -1
    X = X[TRAINING_FEATURES]
    print_debug(X, "predict_with_dataset")
    y_pred = model.predict(X)
    return ticker_data, y_pred

def predict_with_yfinance(ticker: str):
    """Predict volatility using Yahoo Finance data if ticker not in dataset."""
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
# SIDEBAR - Filters
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

filtered_data = data[
    (data["ESG Score"] >= esg_min) & (data["ESG Score"] <= esg_max) &
    (data["Daily_Volatility"] >= vol_min) & (data["Daily_Volatility"] <= vol_max) &
    (data["Year"].isin(year_select)) &
    (data["Ticker"].isin(selected_tickers))
]

# ========================
# TABS WITH DETAILED USAGE EXPLANATION
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
    st.header("Company Ranking & KPIs")
    st.markdown("""
    **Description:** This tab shows key company metrics for selected tickers.
    
    **Contains:** Average ESG Score, Daily Volatility, Adjusted Close price.
    
    **Usage:** Use this tab to quickly see which companies are performing well in ESG and have stable stock prices.
    Hover over rows to analyze specific values. Data is automatically filtered by sidebar settings.
    """)
    kpi_data = filtered_data.groupby("Ticker")[["ESG Score", "Daily_Volatility", "Adj Close"]].mean().sort_values(by=["ESG Score", "Daily_Volatility"], ascending=[False, True])
    st.dataframe(
        kpi_data.style
        .background_gradient(cmap="Greens", subset=["ESG Score"])
        .highlight_max(subset=["Adj Close"], color="lightblue")
    )

# TAB 2
with tab2:
    st.header("ESG Score vs Daily Volatility")
    st.markdown("""
    **Description:** Visual scatter plot of ESG Score vs Daily Volatility.
    
    **Contains:** Each point is a company (Ticker), sized by stock price.
    
    **Usage:** Identify correlations between ESG performance and market volatility. Hover to see ticker details.
    Useful for investors evaluating ESG risk.
    """)
    fig2 = px.scatter(
        filtered_data, x="ESG Score", y="Daily_Volatility",
        color="Ticker", hover_data=["Ticker"],
        color_continuous_scale="Viridis", size="Adj Close"
    )
    st.plotly_chart(fig2, use_container_width=True)

# TAB 3
with tab3:
    st.header("Predict Volatility by Ticker")
    st.markdown("""
    **Description:** Predict the future daily volatility of a company.
    
    **Contains:** Input for ticker symbol, line chart of historical or estimated volatility, predicted latest volatility.
    
    **Usage:** Enter a ticker. If it exists in the dataset, prediction uses internal ESG + market data. Otherwise, fetches last year of stock prices from Yahoo Finance. Helps forecast short-term investment risk.
    """)
    ticker_input = st.text_input("Enter ticker symbol:", "AAPL").upper()
    if ticker_input in data["Ticker"].unique():
        st.success(f"Data for {ticker_input} retrieved from ESG dataset.")
        df_ticker, preds = predict_with_dataset(ticker_input)
        fig_pred = px.line(df_ticker, x="Date", y="Daily_Volatility",
                           title=f"Historical Volatility - {ticker_input}",
                           color_discrete_sequence=["#2E8B57"])
        st.plotly_chart(fig_pred, use_container_width=True)
        st.metric("Predicted Volatility (latest)", f"{preds[-1]:.4f}")
    else:
        st.warning(f"{ticker_input} not in ESG dataset. Using Yahoo Finance data.")
        df_ticker, preds = predict_with_yfinance(ticker_input)
        if df_ticker is not None:
            fig_pred = px.line(df_ticker, x=df_ticker.index, y="Daily_Volatility",
                               title=f"Estimated Volatility - {ticker_input}",
                               color_discrete_sequence=["#3CB371"])
            st.plotly_chart(fig_pred, use_container_width=True)
            st.metric("Predicted Volatility", f"{preds[0]:.4f}")
        else:
            st.error("Unable to retrieve data for this ticker.")

# TAB 4
with tab4:
    st.header("Portfolio Simulation")
    st.markdown("""
    **Description:** Analyze a portfolio composed of selected tickers.
    
    **Contains:** Line chart of each ticker's daily volatility, bar chart of average volatilities.
    
    **Usage:** Select tickers in sidebar. Visualize combined portfolio risk over time. Helps evaluate diversification and ESG-risk trade-offs.
    """)
    if selected_tickers:
        port_data = filtered_data
        fig_port = px.line(
            port_data, x="Date", y="Daily_Volatility",
            color="Ticker", title="Portfolio Volatility Evolution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig_port, use_container_width=True)
        avg_vol = port_data.groupby("Ticker")["Daily_Volatility"].mean()
        st.bar_chart(avg_vol)

# TAB 5
with tab5:
    st.header("Model Performance")
    st.markdown("""
    **Description:** Display predictive model metrics and feature importance.
    
    **Contains:** R², MSE, MAE, feature importance chart.
    
    **Usage:** Evaluate model quality and understand which variables are driving predictions. Helps analysts trust and interpret the model's outputs.
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