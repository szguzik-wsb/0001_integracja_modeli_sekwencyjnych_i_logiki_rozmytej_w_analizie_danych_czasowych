# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Wspolna konfiguracja dla wszystkich eksperymentow.
"""
import os

# Sciezki
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "zewnetrzne")
RESULTS_DIR = os.path.join(BASE_DIR, "wyniki")

# Zbiory danych
DATASETS = {
    "SP500": {
        "file": os.path.join(EXTERNAL_DATA_DIR, "SP500", "sp500_daily.csv"),
        "date_col": "Date",
        "close_col": "Close",
        "name": "S&P 500",
    },
    "WIG20": {
        "file": os.path.join(EXTERNAL_DATA_DIR, "WIG20", "wig20_d.csv"),
        "date_col": "Data",
        "close_col": "Zamkniecie",
        "name": "WIG20",
    },
    "EURUSD": {
        "file": os.path.join(EXTERNAL_DATA_DIR, "EUR_USD", "eurusd_daily.csv"),
        "date_col": "Date",
        "close_col": "Close",
        "name": "EUR/USD",
    },
    "BTCUSD": {
        "file": os.path.join(EXTERNAL_DATA_DIR, "BTC_USD", "btcusd_daily.csv"),
        "date_col": "Date",
        "close_col": "Close",
        "name": "BTC/USD",
    },
    "DAX": {
        "file": os.path.join(EXTERNAL_DATA_DIR, "DAX", "dax_daily.csv"),
        "date_col": "Date",
        "close_col": "Close",
        "name": "DAX",
    },
    "NIKKEI": {
        "file": os.path.join(EXTERNAL_DATA_DIR, "NIKKEI", "nikkei_daily.csv"),
        "date_col": "Date",
        "close_col": "Close",
        "name": "Nikkei 225",
    },
    "GOLD": {
        "file": os.path.join(EXTERNAL_DATA_DIR, "GOLD", "gold_daily.csv"),
        "date_col": "Date",
        "close_col": "Close",
        "name": "Gold",
    },
}

# Podzial danych
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Parametry wspolne
LOOKBACK = 30       # okno wejsciowe (dni)
HORIZON = 1         # horyzont prognozy (1 dzien)
RANDOM_SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
