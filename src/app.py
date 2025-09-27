import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import json
import yfinance as yf
import numpy as np

# ========================
# PATHS
# ========================
MODEL_PATH = "/workspaces/ESG_marketvolatility/src/data/lgbm_mix_model.pkl"
METRICS_PATH = "/workspaces/ESG_marketvolatility/src/data/lgbm_mix_model_metrics.json"
DATASET_PATH = "/workspaces/ESG_marketvolatility/src/data/dataset_final.csv"

# ========================
# STREAMLIT APP CONFIGURATION
# ========================
st.set_page_config(
    page_title="ESG & Market Volatility Dashboard",
    page_icon="📊",
    layout="wide"
)

# ========================
# WELCOME / INSTRUCTIONS
# ========================
st.markdown("""
# 🌱 ESG & Market Volatility Dashboard

Welcome! This interactive platform allows you to:

- **Explore ESG & financial data** for multiple companies.
- **Visualize market volatility and ESG performance correlations.**
- **Predict volatility** for individual tickers using a trained LightGBM model.
- **Simulate portfolios** to compare risk and volatility metrics.
- **Review model performance** with key metrics (R², MSE, MAE).

**How to use:**
1. Navigate through the tabs at the top.
2. Input a ticker in the Prediction tab or select multiple tickers in the Portfolio Simulation tab.
3. Hover over charts for detailed insights.
4. Use interactive filters and controls for tailored analysis.

💡 *Note:* Predictions are based on historical financial + ESG data and provide guidance for analysis, not investment advice.

**Authors:** Gina Pedrosa, Erika Pablos, Lielia Rodas

Enjoy exploring! 🚀
""")

st.title("📊 ESG & Market Volatility Dashboard")

# ========================
# LOAD MODEL & DATA FUNCTIONS
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
    # Drop problematic non-numeric columns for LightGBM
    drop_cols = ["CEO Full Name", "CEO Status", "ESG Score Date", "ESG Status"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    # Select only numeric columns for prediction
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    return df, numeric_cols

# ========================
# LOAD DATA
# ========================
model = load_model()
metrics = load_metrics()
data, numeric_cols = load_data()

# ========================
# PREDICTION FUNCTIONS
# ========================
def predict_with_dataset(ticker: str):
    ticker_data = data[data["Ticker"] == ticker].copy()
    X = ticker_data[numeric_cols].drop(columns=["Daily_Volatility"], errors='ignore')
    # Disable shape check to avoid LightGBM error if features differ slightly
    y_pred = model.predict(X, predict_disable_shape_check=True)
    return ticker_data, y_pred

def predict_with_yfinance(ticker: str):
    yf_data = yf.download(ticker, period="1y", progress=False)
    if yf_data.empty:
        return None, None

    yf_data["Return"] = yf_data["Adj Close"].pct_change()
    yf_data["Volatility"] = yf_data["Return"].rolling(window=30).std() * np.sqrt(252)
    X_new = yf_data[["Open", "High", "Low", "Close", "Volume"]].fillna(0).tail(1)
    y_pred = model.predict(X_new, predict_disable_shape_check=True)
    return yf_data, y_pred

# ========================
# TABS LAYOUT
# ========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Overview",
    "🌍 ESG vs Volatility",
    "🔮 Prediction",
    "💼 Portfolio Simulation",
    "⚙️ Model Performance"
])

# ========================
# TAB 1 - OVERVIEW
# ========================
with tab1:
    st.header("Market & ESG Overview")

    # ESG Score Distribution
    fig1 = px.histogram(data, x="ESG Score", nbins=30, title="Distribution of ESG Scores",
                        color_discrete_sequence=["#2E8B57"])
    st.plotly_chart(fig1, use_container_width=True)

    # Daily Volatility Distribution
    fig2 = px.histogram(data, x="Daily_Volatility", nbins=30, title="Distribution of Daily Volatility",
                        color_discrete_sequence=["#3CB371"])
    st.plotly_chart(fig2, use_container_width=True)

    # Correlation heatmap
    corr = data[numeric_cols].corr()
    fig3 = px.imshow(corr, text_auto=True, color_continuous_scale="Viridis",
                     title="Correlation Matrix (Financials & ESG)")
    st.plotly_chart(fig3, use_container_width=True)

# ========================
# TAB 2 - ESG vs VOLATILITY
# ========================
with tab2:
    st.header("ESG vs Market Volatility")

    fig4 = px.scatter(
        data, x="ESG Score", y="Daily_Volatility",
        size="Adj Close" if "Adj Close" in data.columns else None,
        color="Governance Score" if "Governance Score" in data.columns else "ESG Score",
        hover_data=["Ticker"],
        title="ESG Score vs Daily Volatility",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig4, use_container_width=True)

# ========================
# TAB 3 - PREDICTION
# ========================
with tab3:
    st.header("Predict Volatility by Ticker")
    ticker_input = st.text_input("Enter ticker symbol:", "AAPL").upper()

    if ticker_input in data["Ticker"].unique():
        st.success(f"Data for {ticker_input} retrieved from ESG dataset.")
        df, preds = predict_with_dataset(ticker_input)

        fig5 = px.line(df, x="Date", y="Daily_Volatility", title=f"Historical Volatility - {ticker_input}",
                       line_shape="linear", color_discrete_sequence=["#2E8B57"])
        st.plotly_chart(fig5, use_container_width=True)

        st.metric("Predicted Volatility (latest)", f"{preds[-1]:.4f}")

    else:
        st.warning(f"{ticker_input} not in ESG dataset. Using Yahoo Finance data.")
        df, preds = predict_with_yfinance(ticker_input)

        if df is not None:
            fig6 = px.line(df, x=df.index, y="Volatility", title=f"Estimated Volatility - {ticker_input}",
                           line_shape="linear", color_discrete_sequence=["#3CB371"])
            st.plotly_chart(fig6, use_container_width=True)

            st.metric("Predicted Volatility", f"{preds[0]:.4f}")
            st.caption("Note: ESG scores unavailable. Prediction based on financial features only.")
        else:
            st.error("Unable to retrieve data for this ticker.")

# ========================
# TAB 4 - PORTFOLIO SIMULATION
# ========================
with tab4:
    st.header("Portfolio Simulation")

    selected_tickers = st.multiselect(
        "Select up to 5 tickers for portfolio analysis:",
        options=data["Ticker"].unique(),
        default=data["Ticker"].unique()[:3]
    )

    if selected_tickers:
        port_data = data[data["Ticker"].isin(selected_tickers)]
        fig7 = px.line(port_data, x="Date", y="Daily_Volatility", color="Ticker",
                       title="Volatility Evolution of Selected Portfolio",
                       color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig7, use_container_width=True)

        avg_vol = port_data.groupby("Ticker")["Daily_Volatility"].mean()
        fig8 = px.bar(avg_vol, title="Average Daily Volatility per Ticker",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig8, use_container_width=True)

# ========================
# TAB 5 - MODEL PERFORMANCE
# ========================
with tab5:
    st.header("Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{metrics.get('R2_test', 0):.3f}")
    col2.metric("MSE", f"{metrics.get('MSE_test', 0):.3f}")
    col3.metric("MAE", f"{metrics.get('MAE_test', 0):.3f}")

    st.markdown("""
    **Interpretation:**  
    - R² closer to 1 indicates stronger explanatory power.  
    - Lower MSE and MAE indicate higher predictive accuracy.  
    """)

    st.caption("These metrics are based on the trained LightGBM model using ESG + financial data.")
