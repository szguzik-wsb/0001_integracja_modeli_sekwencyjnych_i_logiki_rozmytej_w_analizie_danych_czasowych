# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 15: Dokladnosc kierunkowa (Directional Accuracy).
Mierzy % poprawnych prognoz kierunku zmiany ceny (wzrost/spadek).
DA jest wazniejsza niz MAPE z perspektywy inwestora.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from utils import save_results
from config import DATASETS

OUT = os.path.dirname(os.path.abspath(__file__))
PRED_DIRS = {
    "ARIMA": os.path.join(os.path.dirname(OUT), "01_arima_baseline"),
    "LSTM": os.path.join(os.path.dirname(OUT), "02_lstm_baseline"),
    "TCN": os.path.join(os.path.dirname(OUT), "03_tcn_baseline"),
    "Mamdani": os.path.join(os.path.dirname(OUT), "04_mamdani_baseline"),
    "TCN-Mamdani": os.path.join(os.path.dirname(OUT), "05_tcn_mamdani_hybrid"),
}


def directional_accuracy(actual, predicted):
    """Procent poprawnych prognoz kierunku zmiany."""
    actual_dir = np.diff(actual)
    pred_dir = predicted[1:] - actual[:-1]  # prognozowana zmiana vs poprzednia wartosc
    correct = np.sign(actual_dir) == np.sign(pred_dir)
    return np.mean(correct) * 100


RESULTS = {}

for ds_key in DATASETS:
    ds_name = DATASETS[ds_key]["name"]
    print(f"\n=== {ds_name} ===")
    ds_results = {}

    for model_name, model_dir in PRED_DIRS.items():
        pred_file = os.path.join(model_dir, f"prognozy_{ds_key}.csv")
        if not os.path.exists(pred_file):
            continue

        df = pd.read_csv(pred_file)
        actual = df["Actual"].values
        predicted = df["Predicted"].values

        # Directional accuracy
        da = directional_accuracy(actual, predicted)

        # Liczba poprawnych up/down
        actual_dir = np.diff(actual)
        pred_dir = predicted[1:] - actual[:-1]
        n_up = np.sum(actual_dir > 0)
        n_down = np.sum(actual_dir <= 0)
        correct_up = np.sum((np.sign(actual_dir) == np.sign(pred_dir)) & (actual_dir > 0))
        correct_down = np.sum((np.sign(actual_dir) == np.sign(pred_dir)) & (actual_dir <= 0))

        da_up = (correct_up / n_up * 100) if n_up > 0 else 0
        da_down = (correct_down / n_down * 100) if n_down > 0 else 0

        ds_results[model_name] = {
            "DA": round(da, 2),
            "DA_up": round(da_up, 2),
            "DA_down": round(da_down, 2),
            "n_up": int(n_up),
            "n_down": int(n_down),
            "correct_up": int(correct_up),
            "correct_down": int(correct_down),
        }
        print(f"  {model_name:15s}: DA={da:.1f}%  (up={da_up:.1f}%, down={da_down:.1f}%)")

    RESULTS[ds_key] = ds_results

# Podsumowanie
print("\n=== PODSUMOWANIE DA [%] ===")
print(f"{'Model':15s}", end="")
for ds in DATASETS:
    print(f"  {DATASETS[ds]['name']:>10s}", end="")
print()
print("-" * 60)
for model in PRED_DIRS:
    print(f"{model:15s}", end="")
    for ds in DATASETS:
        if model in RESULTS.get(ds, {}):
            print(f"  {RESULTS[ds][model]['DA']:>10.1f}", end="")
        else:
            print(f"  {'---':>10s}", end="")
    print()

save_results({"eksperyment": "Directional Accuracy", "wyniki": RESULTS},
             os.path.join(OUT, "wyniki.json"))
print("\n=== EKSPERYMENT 15 ZAKONCZONY ===")
