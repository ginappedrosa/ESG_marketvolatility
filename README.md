 # ESG & Stock market volatility prediction 📊🌱🤖

**Authors**: Gina Pedrosa, Erika Pablos, Lielia Rodas  

---

## 📌 Project overview

This project integrates **financial stock market data** with **ESG (Environmental, Social, and Governance) ratings** to build a set of **machine learning models** capable of predicting **daily stock returns and volatility**.  

The workflow includes:
- **Data exploration & preprocessing** (`project_explore.ipynb`):  
  Cleaning, feature engineering (returns, rolling volatility), ESG data integration, and exploratory data analysis (EDA).  
- **Modeling**:  
  Comparison of different ML approaches (Linear Regression, XGBoost, LightGBM, CatBoost).  
  Best models are saved as `.pkl` for deployment.  
- **Deployment** (`app.ipynb`):  
  A **Streamlit web application** where users can input a ticker and visualize predictions, ESG scores, and volatility forecasts.  

---

## 🎯 Objectives

- **Predict stock behavior**: Use ESG scores and historical stock data to forecast **daily returns** and **volatility**.  
- **Support decision-making**: Provide insights for investors, companies, and stakeholders.  
- **Understand ESG impact**: Analyze the role of ESG performance on stock stability and perception.  
- **Accessible tool**: Through a Streamlit app, make results interpretable and interactive.  

---

## 📊 Dataset description

The file **`dataset_final.csv`** contains the following columns:

| Column           | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| Date             | Trading date of the stock.                                                  |
| Ticker           | Stock symbol of the company.                                                |
| Adj Close        | Adjusted closing price (accounts for splits/dividends).                     |
| Close            | Closing price of the stock.                                                 |
| High             | Highest price during the trading day.                                       |
| Low              | Lowest price during the trading day.                                        |
| Open             | Opening price of the stock.                                                 |
| Volume           | Number of shares traded.                                                    |
| ESG Score        | Overall Environmental, Social, and Governance score.                        |
| Governance Score | Governance performance score.                                               |
| Environment Score| Environmental performance score.                                            |
| Social Score     | Social responsibility performance score.                                    |
| ESG Score Date   | Date when the ESG score was assigned or updated.                            |
| ESG Status       | Current ESG rating status.                                                  |
| CEO Full Name    | Full name of the company's CEO.                                             |
| CEO Gender       | Gender of the CEO.                                                          |
| CEO Status       | Used to identify whether obtaining CEO info was successful.                 |
| Year             | Year of the trading data.                                                   |
| Daily_Return     | Daily % change in adjusted closing price. **(Target for prediction)**        |
| Daily_Volatility | Rolling std of daily returns, measuring stock variability. **(Target)**     |

---

## 📊 Ticker and Company name reference

Below is the updated list of tickers and their corresponding company names, directly extracted from the actual dataset used in the dashboard (`dataset_final.csv`).

| Ticker | Company Name |
|--------|--------------|
| A      | Mr. Michael R. McMullen |
| AAL    | Mr. Robert D. Isom Jr. |
| AAPL   | Mr. Timothy D. Cook |
| ABBV   | Mr. Richard A. Gonzalez |
| ABT    | Mr. Robert B. Ford |
| ACGL   | Mr. Marc  Grandisson |
| ACN    | Ms. Julie T. Spellman Sweet |
| ADBE   | Mr. Shantanu  Narayen |
| ADI    | Mr. Vincent T. Roche |
| ADM    | Mr. Juan Ricardo Luciano |
| ADP    | Ms. Maria  Black |
| ADSK   | Dr. Andrew  Anagnost |
| AEE    | Mr. Martin J. Lyons Jr. |
| AEP    | Ms. Julia A. Sloat |
| AES    | Mr. Andres Ricardo Gluski Weilert |
| AFL    | Mr. Daniel Paul Amos |
| AIG    | Mr. Peter Salvatore Zaffino |
| AIZ    | Mr. Keith Warner Demmings |
| AJG    | Mr. J. Patrick Gallagher Jr. |
| AKAM   | Dr. F. Thomson Leighton |
| ALB    | Mr. Jerry Kent Masters Jr. |
| ALL    | Mr. Thomas Joseph Wilson II |
| ALLE   | Mr. John H. Stone |
| AMAT   | Mr. Gary E. Dickerson |
| AME    | Mr. David A. Zapico |
| AMGN   | Mr. Robert A. Bradway |
| AMP    | Mr. James M. Cracchiolo |
| AMT    | Mr. Thomas A. Bartlett CPA |
| AMZN   | Mr. Andrew R. Jassy |
| ANET   | Ms. Jayshree V. Ullal |
| ANSS   | Dr. Ajei S. Gopal Ph.D. |
| AOS    | Mr. Kevin J. Wheeler |
| APD    | Mr. Seifollah  Ghasemi |
| APH    | Mr. Richard Adam Norwitt |
| APTV   | Mr. Kevin P. Clark |
| ARE    | Mr. Peter M. Moglia |
| ATO    | Mr. John Kevin Akers |
| AVB    | Mr. Benjamin W. Schall |
| AVY    | Mr. Deon M. Stander |
| AWK    | Ms. M. Susan Hardwick |
| AXP    | Mr. Stephen Joseph Squeri |
| AZO    | Mr. William C. Rhodes III |
| BAC    | Mr. Brian Thomas Moynihan |
| BALL   | Mr. Daniel William Fisher |
| BBWI   | Ms. Gina R. Boswell |
| GOOGL  | Mr. Sundar  Pichai |
| LNT    | Mr. John O. Larsen |
| MMM    | Mr. Michael F. Roman |
| MO     | Mr. William F. Gifford Jr. |
| T      | Mr. John T. Stankey |
... (add all tickers from dataset_final.csv as needed) ...



## ⚙️ How to run the dashboard

1. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

2. Launch the dashboard:
  ```bash
  streamlit run src/app.py
  ```

## 🧠 Project pipeline overview

This dashboard uses a machine learning pipeline based on XGBoost to predict market volatility, integrating ESG (Environmental, Social, Governance) scores and financial data. The pipeline includes:
- **StandardScaler** for feature normalization
- **SelectFromModel** for feature selection
- **XGBRegressor** for volatility prediction
- All models and selectors are loaded from the `models/` folder

## 📋 Dashboard tabs & Features

- **Company overview**: Shows ESG and volatility metrics for the selected ticker, with explanations for interpretation.
- **ESG vs volatility**: Comparative analysis between ESG scores and volatility, with user guidance.
- **Prediction**: Predicts volatility for any ticker, including new ones, using the trained model pipeline.
- **Portfolio simulation**: Simulate a portfolio and analyze ESG/volatility impact.
- **Model performance**: Displays model metrics (R², RMSE) and pipeline details.

## ℹ️ Interpreting ESG & volatility

- **ESG Score**: Higher values indicate better environmental, social, and governance practices. Companies with high ESG scores are generally considered more sustainable and responsible.
- **Volatility**: Measures the risk or price fluctuation of a stock. Lower volatility is typically preferred for stable investments, while higher volatility may indicate greater risk and potential reward.

## 🗂️ Data source

All tickers and company names are extracted from the processed dataset (`data/processed/dataset_final.csv`).

---
*Last updated: October 1, 2025*

