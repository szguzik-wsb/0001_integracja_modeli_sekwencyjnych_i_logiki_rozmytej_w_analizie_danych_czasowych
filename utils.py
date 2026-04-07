"""
Wspolne narzedzia: ladowanie danych, metryki, podzial train/val/test.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config import DATASETS, TRAIN_RATIO, VAL_RATIO, LOOKBACK, HORIZON, RANDOM_SEED


def load_dataset(dataset_key):
    """Laduje zbiór danych i zwraca Series z cenami zamkniecia."""
    info = DATASETS[dataset_key]
    df = pd.read_csv(info["file"])

    # Ujednolicenie kolumn
    date_col = info["date_col"]
    close_col = info["close_col"]

    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df[date_col] = df[date_col].dt.tz_localize(None)  # usun timezone
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df[[date_col, close_col]].dropna()
    df.columns = ["Date", "Close"]

    return df


def split_data(df):
    """Dzieli dane na train/val/test (70/15/15)."""
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()

    return train, val, test


def create_sequences(data, lookback=LOOKBACK, horizon=HORIZON):
    """Tworzy sekwencje wejsciowe (X) i etykiety (y) z danych."""
    X, y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback:i + lookback + horizon])
    return np.array(X), np.array(y).squeeze()


def prepare_data(dataset_key, lookback=LOOKBACK):
    """Pelny pipeline: zaladuj, podziel, skaluj, stworz sekwencje."""
    df = load_dataset(dataset_key)
    train_df, val_df, test_df = split_data(df)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df[["Close"]].values).flatten()
    val_scaled = scaler.transform(val_df[["Close"]].values).flatten()
    test_scaled = scaler.transform(test_df[["Close"]].values).flatten()

    X_train, y_train = create_sequences(train_scaled, lookback)
    X_val, y_val = create_sequences(val_scaled, lookback)
    X_test, y_test = create_sequences(test_scaled, lookback)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "dates_test": test_df["Date"].values[lookback:],
    }


# === METRYKI ===

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    """Root Mean Square Error."""
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def diebold_mariano(e1, e2, h=1):
    """
    Test Diebold-Mariano.
    e1, e2: bledy prognoz dwoch modeli.
    Zwraca (statystyka DM, p-value).
    H0: oba modele sa rownie dobre.
    """
    from scipy import stats

    e1, e2 = np.array(e1), np.array(e2)
    d = e1 ** 2 - e2 ** 2  # roznica kwadratow bledow
    n = len(d)
    d_mean = np.mean(d)
    d_var = np.var(d, ddof=1)

    # Autokowariancja dla h-krokowej prognozy
    gamma = []
    for k in range(h):
        gamma.append(np.mean((d[k:] - d_mean) * (d[:n - k] - d_mean)))

    V = (d_var + 2 * sum(gamma)) / n
    if V <= 0:
        return 0.0, 1.0

    dm_stat = d_mean / np.sqrt(V)
    p_value = 2 * stats.t.sf(np.abs(dm_stat), df=n - 1)

    return dm_stat, p_value


def compute_all_metrics(y_true, y_pred):
    """Oblicza wszystkie metryki naraz."""
    return {
        "MAPE": round(mape(y_true, y_pred), 4),
        "RMSE": round(rmse(y_true, y_pred), 6),
        "MAE": round(mae(y_true, y_pred), 6),
    }


def inverse_transform(scaler, values):
    """Odwraca normalizacje MinMaxScaler."""
    return scaler.inverse_transform(values.reshape(-1, 1)).flatten()


def save_results(results_dict, filepath):
    """Zapisuje wyniki do JSON."""
    import json
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2, default=str)
    print(f"Wyniki zapisane: {filepath}")
