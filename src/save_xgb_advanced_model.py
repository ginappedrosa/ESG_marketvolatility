# Save the selected features and model as pkl
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel

# Load dataset
file_path = 'src/data/dataset_final.csv'
df = pd.read_csv(file_path)

# Feature engineering
features = ['High', 'Low', 'Daily_Return', 'Return_5d', 'Return_10d', 'MA_5', 'Vol_5d', 'Vol_10d']
for f in ['Return_5d', 'Return_10d']:
    df[f] = df.groupby('Ticker')['Adj Close'].pct_change(int(f.split('_')[1][:-1]))
for f in ['MA_5', 'MA_10']:
    df[f] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(int(f.split('_')[1])).mean())
for f in ['Vol_5d', 'Vol_10d']:
    df[f] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(int(f.split('_')[1][:-1])).std())

X = df[features].fillna(0)
y = df['Daily_Volatility'].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=5, gamma=0, reg_alpha=0.1,
                    reg_lambda=1, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

with open('src/data/best_model_XGBoost_mix.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('src/data/feature_names_XGBoost_mix.pkl', 'wb') as f:
    pickle.dump(features, f)
