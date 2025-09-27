 # ESG & Stock Market Volatility Prediction 📊🌱🤖

**Authors**: Gina Pedrosa, Erika Pablos, Lielia Rodas  

---

## 📌 Project Overview

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

## 📊 Dataset Description

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

## 🏢 Tickers and Companies

| Ticker | Company Name                           |
|--------|----------------------------------------|
---

## ⚙️ How to Run

1. Install dependencies  
   ```bash
   pip install -r requirements.txt

