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

## STREAMLIT CONFIG
st.set_page_config(
    page_title="ESG & Market Volatility Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

## MAIN PAGE INFO
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



MODEL_PATH = os.path.join("models", "final_xgb_model.pkl")
FEATURE_SELECTOR_PATH = os.path.join("models", "feature_selector.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")
METRICS_PATH = os.path.join("models", "model_metrics.json")
DATASET_PATH = os.path.join("data", "processed", "dataset_final.csv")
CATEGORICAL_RULES_PATH = os.path.join("categorical_rules.json")

## LOADERS
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_feature_selector():
    with open(FEATURE_SELECTOR_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
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
def load_categorical_rules():
    with open(CATEGORICAL_RULES_PATH, "r") as f:
        return json.load(f)

data = load_data()
categorical_rules = load_categorical_rules()
model = load_model()
feature_selector = load_feature_selector()
scaler = load_scaler()
metrics = load_metrics()
data = load_data()
categorical_rules = load_categorical_rules()

## Get feature names from selector
if hasattr(feature_selector, 'get_support'):
    support = feature_selector.get_support()
    if hasattr(feature_selector, 'feature_names_in_'):
        feature_names = list(feature_selector.feature_names_in_[support])
    else:
        if len(support) == len(data.columns):
            feature_names = list(data.columns[support])
        else:
            feature_names = [col for col, keep in zip(data.columns, support) if keep]
else:
    feature_names = feature_selector if isinstance(feature_selector, list) else list(feature_selector)

## CATEGORICAL ENCODING
def apply_categorical_encoding(df, rules):
    for col, mapping in rules.items():
        if col in df.columns:
            df[f"{col}_n"] = df[col].map(mapping).fillna(-1).astype(int)
        else:
            df[f"{col}_n"] = -1
    return df

data = apply_categorical_encoding(data, categorical_rules)

## DEBUG FUNCTION
def print_debug(X, context):
    with open("debug_features.txt", "a") as f:
        f.write(f"\n==== {context} ====\n")
        f.write(f"X.shape: {X.shape}\n")
        f.write(f"X.columns: {list(X.columns)}\n")

## PREDICTION FUNCTIONS

def predict_with_dataset(ticker: str):
    # Always get updated data from Yahoo Finance
    yf_data = yf.download(ticker, period="1y", progress=False)
    if yf_data.empty:
        return None, None
    if "Adj Close" not in yf_data.columns:
        yf_data["Adj Close"] = yf_data["Close"]
    # Feature engineering exactly as in training
    yf_data["Daily_Return"] = yf_data["Adj Close"].pct_change()
    yf_data["Return_5d"] = yf_data["Adj Close"].pct_change(5)
    yf_data["Return_10d"] = yf_data["Adj Close"].pct_change(10)
    yf_data["MA_5"] = yf_data["Adj Close"].rolling(5).mean()
    yf_data["MA_10"] = yf_data["Adj Close"].rolling(10).mean()
    yf_data["Vol_5d"] = yf_data["Adj Close"].rolling(5).std()
    yf_data["Vol_10d"] = yf_data["Adj Close"].rolling(10).std()
    scaler_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else feature_selector.feature_names_in_
    X = pd.DataFrame(index=yf_data.index)
    for col in scaler_features:
        X[col] = yf_data[col].fillna(0) if col in yf_data.columns else 0
    X = X[scaler_features].fillna(0)
    # Scale and select features
    X_scaled = scaler.transform(X)
    X_selected = feature_selector.transform(X_scaled)
    print_debug(pd.DataFrame(X_selected, columns=feature_names), "predict_with_dataset")
    y_pred = model.predict(X_selected)
    return yf_data, y_pred


def predict_with_hybrid(ticker: str):
    # Alias for predict_with_dataset to maintain compatibility
    return predict_with_dataset(ticker)

## SIDEBAR FILTERS
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

## GLOBAL TICKER INPUT
st.sidebar.header("Ticker Input")
ticker_input = st.sidebar.text_input("Enter any US ticker symbol:", "AAPL").upper()
df_ticker, preds = predict_with_hybrid(ticker_input)
if df_ticker is None or df_ticker.empty:
    st.sidebar.error(f"Ticker {ticker_input} not found or cannot fetch data.")
    df_ticker = pd.DataFrame()
    preds = []

## TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    ":office: Company Overview",
    ":earth_africa: ESG vs Volatility",
    ":crystal_ball: Prediction",
    ":briefcase: Portfolio Simulation",
    ":gear: Model Performance"
])

## TAB 1 - Company Overview
with tab1:
    st.header(f"Company Overview - {ticker_input}")
    if not df_ticker.empty:
        esg_score = data[data["Ticker"] == ticker_input]["ESG Score"].mean()
        adj_close = df_ticker["Adj Close"].mean() if "Adj Close" in df_ticker.columns else None
        if "Daily_Volatility" in df_ticker.columns:
            daily_vol = df_ticker["Daily_Volatility"].mean()
        elif preds is not None and len(preds) > 0:
            daily_vol = preds[-1]
        else:
            daily_vol = None
        kpi_data = pd.DataFrame({
            "ESG Score": [esg_score],
            "Daily Volatility": [daily_vol],
            "Adj Close": [adj_close]
        })
        st.dataframe(kpi_data.style.background_gradient(cmap="Greens", subset=["ESG Score"])
                                     .highlight_max(subset=["Adj Close"], color="lightblue"))
        # User comments on ESG and Volatility
        if esg_score < 20:
            st.info(f"ESG Score: {esg_score:.2f} (Excellent ESG risk, lower is better)")
        elif esg_score < 40:
            st.info(f"ESG Score: {esg_score:.2f} (Moderate ESG risk)")
        else:
            st.warning(f"ESG Score: {esg_score:.2f} (High ESG risk)")
        if daily_vol is not None:
            if daily_vol < 0.02:
                st.info(f"Volatility: {daily_vol:.4f} (Low volatility, more stable)")
            elif daily_vol < 0.05:
                st.info(f"Volatility: {daily_vol:.4f} (Moderate volatility)")
            else:
                st.warning(f"Volatility: {daily_vol:.4f} (High volatility, more risky)")

## TAB 2 - ESG Score vs Volatility
with tab2:
    st.header(f"ESG Score vs Average Volatility - {ticker_input}")
    if not df_ticker.empty:
        esg_val = data[data["Ticker"] == ticker_input]["ESG Score"].mean()
        if "Daily_Volatility" in df_ticker.columns:
            vol_mean = df_ticker["Daily_Volatility"].mean()
        elif preds is not None and len(preds) > 0:
            vol_mean = preds[-1]
        else:
            vol_mean = None
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
        # User comments and comparison
        if esg_val < 20:
            st.info(f"ESG Score: {esg_val:.2f} (Excellent ESG risk, lower is better)")
        elif esg_val < 40:
            st.info(f"ESG Score: {esg_val:.2f} (Moderate ESG risk)")
        else:
            st.warning(f"ESG Score: {esg_val:.2f} (High ESG risk)")
        if vol_mean is not None:
            if vol_mean < 0.02:
                st.info(f"Volatility: {vol_mean:.4f} (Low volatility, more stable)")
            elif vol_mean < 0.05:
                st.info(f"Volatility: {vol_mean:.4f} (Moderate volatility)")
            else:
                st.warning(f"Volatility: {vol_mean:.4f} (High volatility, more risky)")
        # Comparison
        if esg_val is not None and vol_mean is not None:
            if esg_val < 20 and vol_mean < 0.02:
                st.success("This company has excellent ESG risk and is very stable (low volatility).")
            elif esg_val > 40 and vol_mean > 0.05:
                st.warning("This company has high ESG risk and is also highly volatile (risky).")
            elif esg_val < 20 and vol_mean > 0.05:
                st.info("Excellent ESG risk but high volatility. Monitor market conditions.")
            elif esg_val > 40 and vol_mean < 0.02:
                st.info("High ESG risk but low volatility. ESG improvements recommended.")

## TAB 3 - Predict Volatility
with tab3:
    st.header(f"Predict Volatility - {ticker_input}")
    if not df_ticker.empty:
        fig_pred = go.Figure()
        if "Daily_Volatility" in df_ticker.columns:
            fig_pred.add_trace(go.Scatter(
                x=df_ticker.index, y=df_ticker["Daily_Volatility"],
                mode="lines", name="Volatility", line=dict(color="#1F77B4", width=3)
            ))
        else:
            # Generate time series of predictions
            scaler_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else feature_selector.feature_names_in_
            X_pred = pd.DataFrame(index=df_ticker.index)
            for col in scaler_features:
                X_pred[col] = df_ticker[col].fillna(0) if col in df_ticker.columns else 0
            X_pred = X_pred[scaler_features].fillna(0)
            X_scaled = scaler.transform(X_pred)
            X_selected = feature_selector.transform(X_scaled)
            y_pred_series = model.predict(X_selected)
            fig_pred.add_trace(go.Scatter(
                x=df_ticker.index, y=y_pred_series,
                mode="lines", name="Predicted Volatility", line=dict(color="#FF7F0E", width=3)
            ))
        st.plotly_chart(fig_pred, use_container_width=True)
        if preds is not None and len(preds) > 0:
            st.metric("Predicted Volatility (latest)", f"{preds[-1]:.4f}")
        # User comments on ESG and Volatility
        esg_score = data[data["Ticker"] == ticker_input]["ESG Score"].mean()
        if esg_score < 20:
            st.info(f"ESG Score: {esg_score:.2f} (Excellent ESG risk, lower is better)")
        elif esg_score < 40:
            st.info(f"ESG Score: {esg_score:.2f} (Moderate ESG risk)")
        else:
            st.warning(f"ESG Score: {esg_score:.2f} (High ESG risk)")
        if preds is not None and len(preds) > 0:
            if preds[-1] < 0.02:
                st.info(f"Predicted Volatility: {preds[-1]:.4f} (Low volatility, more stable)")
            elif preds[-1] < 0.05:
                st.info(f"Predicted Volatility: {preds[-1]:.4f} (Moderate volatility)")
            else:
                st.warning(f"Predicted Volatility: {preds[-1]:.4f} (High volatility, more risky)")

## TAB 4 - Portfolio Simulation
with tab4:
    st.header("Portfolio Simulation")
    if selected_tickers:
        port_data = pd.DataFrame()
        plot_data = []
        for t in selected_tickers:
            df_t, preds_t = predict_with_hybrid(t)
            if df_t is not None:
                df_t["Ticker"] = t
                if "Daily_Volatility" in df_t.columns:
                    plot_data.append({"x": df_t.index, "y": df_t["Daily_Volatility"], "name": t})
                else:
                    scaler_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else feature_selector.feature_names_in_
                    X_pred = pd.DataFrame(index=df_t.index)
                    for col in scaler_features:
                        X_pred[col] = df_t[col].fillna(0) if col in df_t.columns else 0
                    X_pred = X_pred[scaler_features].fillna(0)
                    X_scaled = scaler.transform(X_pred)
                    X_selected = feature_selector.transform(X_scaled)
                    y_pred_series = model.predict(X_selected)
                    plot_data.append({"x": df_t.index, "y": y_pred_series, "name": t})
                # User comments for each ticker
                esg_score = data[data["Ticker"] == t]["ESG Score"].mean()
                if esg_score < 20:
                    st.info(f"{t} ESG Score: {esg_score:.2f} (Excellent ESG risk, lower is better)")
                elif esg_score < 40:
                    st.info(f"{t} ESG Score: {esg_score:.2f} (Moderate ESG risk)")
                else:
                    st.warning(f"{t} ESG Score: {esg_score:.2f} (High ESG risk)")
                if preds_t is not None and len(preds_t) > 0:
                    if preds_t[-1] < 0.02:
                        st.info(f"{t} Predicted Volatility: {preds_t[-1]:.4f} (Low volatility, more stable)")
                    elif preds_t[-1] < 0.05:
                        st.info(f"{t} Predicted Volatility: {preds_t[-1]:.4f} (Moderate volatility)")
                    else:
                        st.warning(f"{t} Predicted Volatility: {preds_t[-1]:.4f} (High volatility, more risky)")
        if plot_data:
            fig_port = go.Figure()
            for trace in plot_data:
                fig_port.add_trace(go.Scatter(
                    x=trace["x"], y=trace["y"], mode="lines", name=trace["name"]
                ))
            fig_port.update_layout(title="Portfolio Volatility Evolution")
            st.plotly_chart(fig_port, use_container_width=True)
        else:
            st.write("No volatility data available for selected tickers.")


## TAB 5 - Model Performance
with tab5:
    st.header("Model Performance - Final XGBoost Model")
    st.markdown(f"""
    **Model:** Final XGBoost (feature selection applied)

    - **Selected features:** {', '.join(feature_names)}
    - **Test R²:** {metrics.get('r2_test', 0):.3f}
    - **Test RMSE:** {metrics.get('rmse_test', 0):.4f}

    The model predicts daily volatility using selected financial features. Feature selection was performed using XGBoost importance, and only the most predictive variables were retained. This approach improves generalization and avoids overfitting.

    **Interpretation:**
    - A higher predicted volatility means the stock is expected to experience larger price swings in the near future.
    - The R² value indicates the proportion of variance explained by the model: {metrics.get('r2_test', 0)*100:.1f}% in test. The RMSE value shows the average prediction error.
    - The model is robust and realistic, with no data leakage or artificial features.
    """)







