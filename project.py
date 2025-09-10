# 🌸 IMPORTS
import pandas as pd
import yfinance as yf
import numpy as np
from itertools import product
from tqdm import tqdm

# 🔹 1. CARGAR CSV ESG
csv_file = "sp500_esg_info-filtered.csv"  # Asegúrate de subirlo a Colab
esg_df = pd.read_csv(csv_file)

# Revisamos columnas
print("Columnas CSV ESG:", esg_df.columns)
print("Número de tickers en CSV:", esg_df.shape[0])

# 🔹 2. SELECCIONAR LOS 100 TICKERS PRINCIPALES PARA EL EJEMPLO
tickers = esg_df['Ticker'].dropna().unique()[:10]

# 🔹 3. DESCARGAR DATOS FINANCIEROS ÚLTIMOS 5 AÑOS
start_date = "2018-01-01"
end_date = "2023-12-31"

# Diccionario para almacenar los datos
financial_data = {}

# Descarga con tqdm para ver progreso
for t in tqdm(tickers, desc="Descargando datos yfinance"):
    try:
        df = yf.download(t, start=start_date, end=end_date, progress=False)
        if not df.empty:
            df['Ticker'] = t
            financial_data[t] = df
    except Exception as e:
        print(f"Error {t}: {e}")

# Concatenamos todos los datos financieros
fin_df = pd.concat(financial_data.values(), axis=0).reset_index()
print("Datos financieros descargados:", fin_df.shape)

# 🔹 4. UNIFICAR ESG + FINANZAS
# Creamos un DataFrame combinando cada fila financiera con los datos ESG del ticker
final_rows = []
for idx, row in fin_df.iterrows():
    ticker = row['Ticker']
    esg_info = esg_df[esg_df['Ticker'] == ticker]
    if not esg_info.empty:
        esg_data = esg_info.iloc[0].to_dict()
        combined = {**row.to_dict(), **esg_data}
        final_rows.append(combined)

dataset_final = pd.DataFrame(final_rows)
print("Dataset combinado:", dataset_final.shape)

# 🔹 5. CREAR NUEVAS FEATURES (Opcional, para ML)
# Por ejemplo: volatilidad diaria, retorno diario
dataset_final['Daily_Return'] = dataset_final['Adj Close'].pct_change()
dataset_final['Daily_Volatility'] = dataset_final['Daily_Return'].rolling(5).std()

# 🔹 6. LIMPIEZA
dataset_final = dataset_final.dropna()
print("Dataset final limpio:", dataset_final.shape)

# 🔹 7. GUARDAR DATASET
dataset_final.to_csv("dataset_final_completo.csv", index=False)
print("✅ Dataset creado y guardado: dataset_final_completo.csv")

# 🔹 8. INFORMACIÓN FINAL
print(dataset_final.head())