import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel

# Load dataset
file_path = 'src/data/dataset_final.csv'
df = pd.read_csv(file_path)

# Create new financial features
df['Return_5d'] = df.groupby('Ticker')['Adj Close'].pct_change(5)
df['Return_10d'] = df.groupby('Ticker')['Adj Close'].pct_change(10)
df['MA_5'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(5).mean())
df['MA_10'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(10).mean())
df['Vol_5d'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(5).std())
df['Vol_10d'] = df.groupby('Ticker')['Adj Close'].transform(lambda x: x.rolling(10).std())

# Select candidate features (financial + ESG Score)
candidate_features = [
    'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'Year', 'Daily_Return',
    'Return_5d', 'Return_10d', 'MA_5', 'MA_10', 'Vol_5d', 'Vol_10d', 'ESG Score'
]
X = df[candidate_features].fillna(0)
y = df['Daily_Volatility'].fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initial model for feature selection
selector_model = XGBRegressor(n_estimators=200, max_depth=4, random_state=42, n_jobs=-1)
selector_model.fit(X_train_scaled, y_train)
selector = SelectFromModel(selector_model, threshold='median', prefit=True)
X_train_sel = selector.transform(X_train_scaled)
X_test_sel = selector.transform(X_test_scaled)
selected_features = X.columns[selector.get_support()].tolist()
print('Selected features:', selected_features)

# Final model with selected features and hyperparameter tuning
final_model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, gamma=0, reg_alpha=0.1,
                          reg_lambda=1, random_state=42, n_jobs=-1)
final_model.fit(X_train_sel, y_train)
y_pred_test = final_model.predict(X_test_sel)

y_pred_train = final_model.predict(X_train_sel)
r2_train = r2_score(y_train, y_pred_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
print(f'R2 test: {r2_test:.4f}')
print(f'RMSE test: {rmse_test:.4f}')
print(f'R2 train: {r2_train:.4f}')
print(f'RMSE train: {rmse_train:.4f}')
