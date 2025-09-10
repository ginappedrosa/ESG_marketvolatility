import pandas as pd
import yfinance as yf
from tqdm import tqdm
import time

# --- CONFIGURACIÓN ---
esg_csv = '/workspaces/ginappedrosa_project_test/sp500_esg_ceo_info-filtered.csv'   # tu CSV ESG
output_csv = 'dataset_final_completo.csv'
start_date = '2018-01-01'
end_date = '2024-09-01'
chunk_size = 50  # cantidad de tickers por batch

# --- CARGAR ESG ---
esg_df = pd.read_csv(esg_csv)
tickers = esg_df['Ticker'].str.upper().str.strip().tolist()

# --- FUNCIÓN PARA DESCARGAR DATOS ---
def download_tickers_batch(ticker_list):
    batch_data = []
    for t in tqdm(ticker_list, desc="Descargando batch"):
        try:
            df = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if not df.empty:
                df.reset_index(inplace=True)
                df['Ticker'] = t
                batch_data.append(df)
            time.sleep(0.1)  # evitar rate limit
        except Exception as e:
            print(f"Error con {t}: {e}")
    if batch_data:
        return pd.concat(batch_data, ignore_index=True)
    else:
        return pd.DataFrame()

# --- DESCARGAR EN BATCHES ---
all_data = []
for i in range(0, len(tickers), chunk_size):
    batch = tickers[i:i+chunk_size]
    batch_df = download_tickers_batch(batch)
    if not batch_df.empty:
        all_data.append(batch_df)
    print(f"Batch {i//chunk_size + 1} completado, filas acumuladas: {sum([len(d) for d in all_data])}")

# --- CONCATENAR TODOS LOS DATOS ---
if all_data:
    yf_df = pd.concat(all_data, ignore_index=True)
else:
    raise ValueError("No se descargó ningún dato de yfinance.")

# --- MERGE CON ESG ---
final_df = yf_df.merge(esg_df, on='Ticker', how='left')

# --- GUARDAR CSV FINAL ---
final_df.to_csv(output_csv, index=False)
print(f"Dataset final creado: {output_csv}")
print(f"Shape final: {final_df.shape}")
