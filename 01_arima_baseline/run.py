# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""Eksperyment 01: ARIMA baseline na 4 zbiorach danych."""
import sys, os, time, json, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from utils import load_dataset, split_data, compute_all_metrics, save_results
from config import DATASETS

RESULTS = {}

for ds_key, ds_info in DATASETS.items():
    print(f"\n=== {ds_info['name']} ===")
    t0 = time.time()

    df = load_dataset(ds_key)
    train_df, val_df, test_df = split_data(df)

    train_vals = train_df["Close"].values
    test_vals = test_df["Close"].values

    # Dopasuj ARIMA na train, prognozuj test krok po kroku (walk-forward)
    history = list(train_vals)
    history.extend(list(val_df["Close"].values))  # train + val jako historia
    predictions = []

    for i in range(len(test_vals)):
        try:
            model = ARIMA(history, order=(5, 1, 0))
            fitted = model.fit()
            yhat = fitted.forecast(steps=1)[0]
        except Exception:
            yhat = history[-1]  # fallback: ostatnia wartosc

        predictions.append(yhat)
        history.append(test_vals[i])

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_vals)}")

    predictions = np.array(predictions)
    elapsed = time.time() - t0

    metrics = compute_all_metrics(test_vals, predictions)
    metrics["czas_s"] = round(elapsed, 2)
    RESULTS[ds_key] = metrics

    print(f"  MAPE: {metrics['MAPE']}%")
    print(f"  RMSE: {metrics['RMSE']}")
    print(f"  MAE:  {metrics['MAE']}")
    print(f"  Czas: {metrics['czas_s']}s")

    # Zapisz prognozy
    pred_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "Actual": test_vals,
        "Predicted": predictions
    })
    pred_df.to_csv(os.path.join(os.path.dirname(__file__), f"prognozy_{ds_key}.csv"), index=False)

# Zapisz wyniki
save_results({"model": "ARIMA(5,1,0)", "wyniki": RESULTS},
             os.path.join(os.path.dirname(__file__), "wyniki.json"))
print("\n=== ARIMA BASELINE ZAKONCZONE ===")
