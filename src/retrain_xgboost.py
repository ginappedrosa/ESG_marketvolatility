import pandas as pd
import joblib
import json
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Cargar dataset
DATASET_PATH = "src/data/dataset_final.csv"
CATEGORICAL_RULES_PATH = "src/data/categorical_rules.json"
MODEL_PATH = "src/data/best_model_XGBoost_mix.pkl"
METRICS_PATH = "src/data/metrics_XGBoost_mix.json"

def apply_categorical_encoding(df, rules):
    for col, mapping in rules.items():
        if col in df.columns:
            df[f"{col}_n"] = df[col].map(mapping).fillna(-1).astype(int)
        else:
            df[f"{col}_n"] = -1
    return df

# Cargar datos
categorical_rules = json.load(open(CATEGORICAL_RULES_PATH))
df = pd.read_csv(DATASET_PATH)
df = df.drop(columns=["CEO Full Name", "CEO Status"], errors="ignore")
df = apply_categorical_encoding(df, categorical_rules)

TRAINING_FEATURES = [
    "Open","High","Low","Close","Adj Close","Volume",
    "ESG Score","Governance Score","Environment Score","Social Score",
    "Year","Daily_Return"
] + [f"{col}_n" for col in categorical_rules.keys()] + ["DUMMY_FILL"]

# Rellenar DUMMY_FILL si no existe
if "DUMMY_FILL" not in df.columns:
    df["DUMMY_FILL"] = -1

# Eliminar filas con NaN en features
X = df[TRAINING_FEATURES].fillna(-1)
y = df["Daily_Volatility"].fillna(-1)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Guardar modelo
joblib.dump(model, MODEL_PATH)

# Métricas
y_pred = model.predict(X_test)
metrics = {
    "r2_test": r2_score(y_test, y_pred),
    "rmse_test": mean_squared_error(y_test, y_pred) ** 0.5,
    "mae_test": mean_absolute_error(y_test, y_pred)
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print("Modelo y métricas guardados correctamente.")
