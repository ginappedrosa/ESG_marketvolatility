ESG & Stock market volatility prediction 📊🌱🤖

Authors: Gina Pedrosa, Erika Pablos, Lielia Rodas

Project overview:

This project combines financial stock market data with ESG (Environmental, Social, and Governance) ratings to create a machine learning model that predicts stock behavior.

The goal is to provide a comprehensive dataset and predictive model that can be used by:

Investors: To make informed decisions by predicting stock returns and volatility alongside ESG performance, aiming for more stable and responsible investments.

Job seekers or stakeholders: To assess companies with strong ESG values, promoting transparency and confidence in corporate behavior.

Companies themselves: To understand how ESG initiatives might influence stock stability and consider whether investing more in ESG could reduce volatility and improve market perception.

The dataset includes daily stock prices, ESG scores, and derived features like daily returns and volatility, which serve as the target variables for prediction.

Dataset description:

The dataset_final.csv contains the following columns:

| Column                | Description                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------- |
| **Date**              | Trading date of the stock.                                                                              |
| **Ticker**            | Stock symbol of the company.                                                                            |
| **Adj Close**         | Adjusted closing price (accounts for splits/dividends).                                                 |
| **Close**             | Closing price of the stock.                                                                             |
| **High**              | Highest price during the trading day.                                                                   |
| **Low**               | Lowest price during the trading day.                                                                    |
| **Open**              | Opening price of the stock.                                                                             |
| **Volume**            | Number of shares traded.                                                                                |
| **ESG Score**         | Overall Environmental, Social, and Governance score.                                                    |
| **Governance Score**  | Governance performance score.                                                                           |
| **Environment Score** | Environmental performance score.                                                                        |
| **Social Score**      | Social responsibility performance score.                                                                |
| **ESG Score Date**    | Date when the ESG score was assigned or updated.                                                        |
| **ESG Status**        | Current ESG rating status.                                              |
| **CEO Full Name**     | Full name of the company's CEO.                                                                         |
| **CEO Gender**        | Gender of the CEO.                                                                                      |
| **CEO Status**        | This is used to identify whether obtaining CEO info was succesful                                                              |
| **Year**              | Year of the trading data.                                                                               |
| **Daily\_Return**     | Daily percentage change in adjusted closing price. (Target for prediction)                              |
| **Daily\_Volatility** | Rolling standard deviation of daily returns, measuring stock price variability. (Target for prediction) |


Tickers and corresponding companies:

| Ticker | Company Name                          |
| ------ | ------------------------------------- |
| A      | Agilent Technologies Inc.             |
| AAL    | American Airlines Group Inc.          |
| AAPL   | Apple Inc.                            |
| ABBV   | AbbVie Inc.                           |
| ABT    | Abbott Laboratories                   |
| ACGL   | Arch Capital Group Ltd.               |
| ACN    | Accenture plc                         |
| ADBE   | Adobe Inc.                            |
| ADI    | Analog Devices, Inc.                  |
| ADM    | Archer-Daniels-Midland Company        |
| ADP    | Automatic Data Processing, Inc.       |
| ADSK   | Autodesk, Inc.                        |
| AEE    | Ameren Corporation                    |
| AEP    | American Electric Power Company, Inc. |
| AES    | AES Corporation                       |
| AFL    | Aflac Incorporated                    |
| AIG    | American International Group, Inc.    |
| AIZ    | Assurant, Inc.                        |
| AJG    | Arthur J. Gallagher & Co.             |
| AKAM   | Akamai Technologies, Inc.             |
| ALB    | Albemarle Corporation                 |
| ALL    | The Allstate Corporation              |
| ALLE   | Allegion plc                          |
| AMAT   | Applied Materials, Inc.               |
| AME    | A. O. Smith Corporation               |
| AMGN   | Amgen Inc.                            |
| AMP    | Ameriprise Financial, Inc.            |
| AMT    | American Tower Corporation            |
| AMZN   | Amazon.com, Inc.                      |
| ANET   | Arista Networks, Inc.                 |
| ANSS   | ANSYS, Inc.                           |
| AOS    | A. O. Smith Corporation               |
| APD    | Air Products and Chemicals, Inc.      |
| APH    | Amphenol Corporation                  |
| APTV   | Aptiv PLC                             |
| ARE    | Alexandria Real Estate Equities, Inc. |
| ATO    | Atmos Energy Corporation              |
| AVB    | AvalonBay Communities, Inc.           |
| AVY    | Avery Dennison Corporation            |
| AWK    | American Water Works Company, Inc.    |
| AXP    | American Express Company              |
| AZO    | AutoZone, Inc.                        |
| BAC    | Bank of America Corporation           |
| BALL   | Ball Corporation                      |
| BBWI   | Bath & Body Works, Inc.               |
| GOOGL  | Alphabet Inc. (Class A)               |
| LNT    | Alliant Energy Corporation            |
| MMM    | 3M Company                            |
| MO     | Altria Group, Inc.                    |
| T      | AT\&T Inc.                            |

Project structure:

project/
│
├── dataset_final.csv       # Final dataset with stock prices and ESG scores
├── data_preprocessing.py   # Script to clean and prepare the data
├── model_training.py       # Machine Learning model for predicting returns/volatility
├── analysis.ipynb          # Exploratory data analysis and feature engineering
├── README.md               # Project description and dataset details
└── requirements.txt        # Python dependencies

Objective

This project aims to:

Predict stock behavior: Use ESG scores and historical stock data to forecast daily returns and volatility.

Support decision-making: Help investors, companies, and stakeholders make better-informed decisions using predictive insights.

Understand ESG impact: Analyze how ESG performance affects stock stability and market behavior.

Provide clarity for everyone: Whether you’re a professional investor, a student, or a job seeker, the dataset helps understand companies’ performance and values.

Authors:

Gina Pedrosa

Erika Pablos

Lielia Rodas
