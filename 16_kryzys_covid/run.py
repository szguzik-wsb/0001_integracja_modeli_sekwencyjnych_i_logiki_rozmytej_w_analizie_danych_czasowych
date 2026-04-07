"""
Eksperyment 16: Test na okresach kryzysowych.
Wydziela okresy wysokiej zmiennosci (COVID crash 2020, bear market 2022)
i porownuje modele w tych trudnych warunkach.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from utils import load_dataset, split_data, mape, rmse, mae, save_results
from config import DATASETS

OUT = os.path.dirname(os.path.abspath(__file__))
PRED_DIRS = {
    "ARIMA": os.path.join(os.path.dirname(OUT), "01_arima_baseline"),
    "LSTM": os.path.join(os.path.dirname(OUT), "02_lstm_baseline"),
    "TCN": os.path.join(os.path.dirname(OUT), "03_tcn_baseline"),
    "Mamdani": os.path.join(os.path.dirname(OUT), "04_mamdani_baseline"),
    "TCN-Mamdani": os.path.join(os.path.dirname(OUT), "05_tcn_mamdani_hybrid"),
}

# Okresy kryzysowe w S&P 500 (daty orientacyjne w zbiorze testowym)
# Test set zaczyna sie ok 2023-06 (70%+15% z 2015-2025)
# Wiec COVID (2020) i bear 2022 sa w TRAIN/VAL, nie w TEST
# Musimy uzyc CALEGO zbioru i przetrenowac na danych DO kryzysu

# Alternatywa: analizujemy zmiennosc w zbiorze testowym
# i dzielimy test na "spokojne" vs "niespokojne" dni

RESULTS = {}

for ds_key in ["SP500", "WIG20", "BTCUSD"]:
    ds_name = DATASETS[ds_key]["name"]
    print(f"\n=== {ds_name} ===")

    ds_results = {"spokojna": {}, "niestabilna": {}}

    for model_name, model_dir in PRED_DIRS.items():
        pred_file = os.path.join(model_dir, f"prognozy_{ds_key}.csv")
        if not os.path.exists(pred_file):
            continue

        df = pd.read_csv(pred_file)
        actual = df["Actual"].values
        predicted = df["Predicted"].values

        # Oblicz dzienne zwroty
        returns = np.abs(np.diff(actual) / actual[:-1]) * 100
        # Prog: mediana zwrotow = "spokojna", powyzej 90 percentyla = "niestabilna"
        threshold = np.percentile(returns, 75)

        calm_mask = returns <= threshold
        volatile_mask = returns > threshold

        # Metryki na spokojnych dniach
        if np.sum(calm_mask) > 10:
            calm_actual = actual[1:][calm_mask]
            calm_pred = predicted[1:][calm_mask]
            ds_results["spokojna"][model_name] = {
                "MAPE": round(mape(calm_actual, calm_pred), 4),
                "RMSE": round(rmse(calm_actual, calm_pred), 4),
                "n": int(np.sum(calm_mask)),
            }

        # Metryki na niestabilnych dniach
        if np.sum(volatile_mask) > 5:
            vol_actual = actual[1:][volatile_mask]
            vol_pred = predicted[1:][volatile_mask]
            ds_results["niestabilna"][model_name] = {
                "MAPE": round(mape(vol_actual, vol_pred), 4),
                "RMSE": round(rmse(vol_actual, vol_pred), 4),
                "n": int(np.sum(volatile_mask)),
            }

    RESULTS[ds_key] = ds_results

    print(f"  Prog zmiennosci (75 percentyl): {threshold:.2f}%")
    print(f"\n  SPOKOJNA ({ds_results['spokojna'].get('ARIMA', {}).get('n', '?')} dni):")
    for m in PRED_DIRS:
        if m in ds_results["spokojna"]:
            print(f"    {m:15s}: MAPE={ds_results['spokojna'][m]['MAPE']:.3f}%")

    print(f"\n  NIESTABILNA ({ds_results['niestabilna'].get('ARIMA', {}).get('n', '?')} dni):")
    for m in PRED_DIRS:
        if m in ds_results["niestabilna"]:
            print(f"    {m:15s}: MAPE={ds_results['niestabilna'][m]['MAPE']:.3f}%")

save_results({"eksperyment": "Analiza reżimów zmienności", "wyniki": RESULTS},
             os.path.join(OUT, "wyniki.json"))
print("\n=== EKSPERYMENT 16 ZAKONCZONY ===")
