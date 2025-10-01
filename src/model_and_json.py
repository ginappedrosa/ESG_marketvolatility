import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import json

# Load data
df = pd.read_csv('data/processed/dataset_final.csv')

# Feature engineering (same as in the notebook)
df['Return_5d'] = df.groupby('Ticker')['Adj Close'].pct_change(5)
df['Return_10d'] = df.groupby('Ticker')['Adj Close'].pct_change(10)
df['MA_5'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(5).mean())
df['MA_10'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(10).mean())
df['Vol_5d'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(5).std())
df['Vol_10d'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(10).std())

candidate_features = [
    'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'Year', 'Daily_Return',
    'Return_5d', 'Return_10d', 'MA_5', 'MA_10', 'Vol_5d', 'Vol_10d', 'ESG Score'
]
X = df[candidate_features].fillna(0)
y = df['Daily_Volatility'].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

selector_model = XGBRegressor(n_estimators=200, max_depth=4, random_state=42, n_jobs=-1)
selector_model.fit(X_train_scaled, y_train)
selector = SelectFromModel(selector_model, threshold='median', prefit=True)
X_train_sel = selector.transform(X_train_scaled)
X_test_sel = selector.transform(X_test_scaled)

final_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, gamma=0, reg_alpha=0.1,
                          reg_lambda=1, random_state=42, n_jobs=-1)
final_model.fit(X_train_sel, y_train)

# Predictions and metrics
y_pred_test = final_model.predict(X_test_sel)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Save model, selector, and scaler with pickle
with open('models/final_xgb_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
with open('models/feature_selector.pkl', 'wb') as f:
    pickle.dump(selector, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save metrics to JSON
metrics = {
    'r2_test': r2_test,
    'rmse_test': rmse_test
}
with open('models/model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("Model, selector, scaler, and metrics saved in the models folder/")
