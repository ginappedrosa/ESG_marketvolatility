import pandas as pd
import yfinance as yf
from tqdm import tqdm

# CSV de Sustainalytics ESG
esg_csv = "/workspaces/ginappedrosa_project_test/sp500_esg_ceo_info-filtered.csv"  # asegúrate de que está en tu carpeta
esg_df = pd.read_csv(esg_csv)

print("Columnas CSV ESG:", esg_df.columns)
print("Número de tickers en CSV:", esg_df["Ticker"].nunique())

# Seleccionamos solo 50 tickers de prueba
tickers = esg_df["Ticker"].dropna().unique().tolist()[:50]

start_date = "2018-01-01"
end_date = "2023-12-31"

all_data = []

# Descargar datos de 50 en 50 (batch)
batch_size = 50
for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    print(f"\nDescargando batch {i//batch_size + 1} de {len(tickers)//batch_size + 1}...")
    
    try:
        df = yf.download(
            batch,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False  # 🌸 aseguramos que baje "Adj Close"
        )
        
        # Pasamos de columnas multi-índice a columnas simples
        df = df.stack(level=1).reset_index()
        df.rename(columns={"level_1": "Ticker"}, inplace=True)
        
        all_data.append(df)
    except Exception as e:
        print(f"⚠️ Error en batch {i//batch_size + 1}: {e}")

# Unimos todos los datos financieros
fin_df = pd.concat(all_data, ignore_index=True)

print(f"\n✅ Datos financieros descargados: {fin_df.shape}")

# Unimos con ESG (por ticker)
dataset_final = pd.merge(fin_df, esg_df, on="Ticker", how="inner")

# Creamos features de volatilidad
dataset_final["Daily_Return"] = dataset_final.groupby("Ticker")["Adj Close"].pct_change()
dataset_final["Daily_Volatility"] = (
    dataset_final.groupby("Ticker")["Daily_Return"]
    .rolling(5)
    .std()
    .reset_index(0, drop=True)
)

# Guardar CSV final
dataset_final.to_csv("dataset_final.csv", index=False)

print(f"\n🌸 Dataset guardado como 'dataset_final.csv'")
print("Shape final:", dataset_final.shape)
print("\nPrimeras filas:\n", dataset_final.head())

dataset_final.info()
