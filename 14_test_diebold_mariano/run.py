"""
Eksperyment 14: Test istotnosci statystycznej Diebold-Mariano.

Porownanie TCN+Mamdani vs kazdy baseline (ARIMA, LSTM, TCN, Mamdani)
na wszystkich 4 zbiorach danych.

H0: Oba modele sa rownie dobre (brak roznic w dokladnosci prognoz).
H1: Modele roznia sie istotnie w dokladnosci prognoz.
Poziom istotnosci: alpha = 0.05.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from utils import diebold_mariano, save_results
from config import DATASETS

ALPHA = 0.05
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mapowanie nazw eksperymentow na foldery
EXPERIMENT_DIRS = {
    "ARIMA": os.path.join(BASE_DIR, "01_arima_baseline"),
    "LSTM": os.path.join(BASE_DIR, "02_lstm_baseline"),
    "TCN": os.path.join(BASE_DIR, "03_tcn_baseline"),
    "Mamdani": os.path.join(BASE_DIR, "04_mamdani_baseline"),
    "TCN_Mamdani": os.path.join(BASE_DIR, "05_tcn_mamdani_hybrid"),
}

BASELINES = ["ARIMA", "LSTM", "TCN", "Mamdani"]
HYBRID = "TCN_Mamdani"
DATASET_KEYS = ["SP500", "WIG20", "EURUSD", "BTCUSD"]


def load_predictions(experiment_name, dataset_key):
    """Laduje prognozy z pliku CSV danego eksperymentu."""
    exp_dir = EXPERIMENT_DIRS[experiment_name]
    csv_path = os.path.join(exp_dir, f"prognozy_{dataset_key}.csv")

    if not os.path.exists(csv_path):
        print(f"  BRAK PLIKU: {csv_path}")
        return None, None

    df = pd.read_csv(csv_path)

    if "Actual" in df.columns and "Predicted" in df.columns:
        return df["Actual"].values, df["Predicted"].values
    else:
        print(f"  BLEDNE KOLUMNY w {csv_path}: {list(df.columns)}")
        return None, None


print("=== Eksperyment 14: Test Diebold-Mariano ===")
print(f"  Poziom istotnosci: alpha = {ALPHA}")
print(f"  H0: Oba modele sa rownie dobre")
print(f"  Porownanie: TCN+Mamdani vs {BASELINES}\n")

RESULTS = {}
summary_table = []

for ds_key in DATASET_KEYS:
    ds_name = DATASETS[ds_key]["name"]
    print(f"\n=== {ds_name} ({ds_key}) ===")

    # Zaladuj prognozy modelu hybrydowego
    actual_hybrid, pred_hybrid = load_predictions(HYBRID, ds_key)
    if actual_hybrid is None:
        print(f"  POMINIETO: brak prognoz TCN+Mamdani dla {ds_key}")
        continue

    errors_hybrid = actual_hybrid - pred_hybrid

    ds_results = {}

    for baseline in BASELINES:
        actual_bl, pred_bl = load_predictions(baseline, ds_key)

        if actual_bl is None:
            print(f"  POMINIETO: brak prognoz {baseline} dla {ds_key}")
            ds_results[baseline] = {
                "status": "brak_danych",
                "DM_statystyka": None,
                "p_value": None,
                "istotne": None,
            }
            continue

        # Wyrownanie dlugosci (rozne modele moga miec rozne dlugosci prognoz)
        n_min = min(len(errors_hybrid), len(actual_bl))
        errors_bl = actual_bl[:n_min] - pred_bl[:n_min]
        errors_h = errors_hybrid[:n_min]

        # Test Diebold-Mariano
        dm_stat, p_value = diebold_mariano(errors_bl, errors_h, h=1)

        # Interpretacja
        is_significant = p_value < ALPHA
        if is_significant:
            if dm_stat > 0:
                interpretation = "TCN+Mamdani LEPSZY (istotnie statystycznie)"
            else:
                interpretation = "Baseline LEPSZY (istotnie statystycznie)"
        else:
            interpretation = "BRAK istotnej roznicy"

        ds_results[baseline] = {
            "DM_statystyka": round(float(dm_stat), 6),
            "p_value": round(float(p_value), 6),
            "istotne": is_significant,
            "interpretacja": interpretation,
            "n_probek": n_min,
            "alpha": ALPHA,
        }

        sig_marker = "*" if is_significant else " "
        print(f"  vs {baseline:<10}: DM={dm_stat:>8.4f}, p={p_value:.6f} {sig_marker} {interpretation}")

        summary_table.append({
            "dataset": ds_key,
            "baseline": baseline,
            "DM_stat": round(float(dm_stat), 4),
            "p_value": round(float(p_value), 6),
            "istotne": is_significant,
            "interpretacja": interpretation,
        })

    RESULTS[ds_key] = ds_results

# Podsumowanie
print("\n\n=== PODSUMOWANIE TESTU DIEBOLD-MARIANO ===")
print(f"{'Dataset':<10} {'vs Baseline':<12} {'DM stat':>10} {'p-value':>10} {'Istotne':>10} {'Interpretacja'}")
print("-" * 90)
for row in summary_table:
    sig = "TAK" if row["istotne"] else "NIE"
    print(f"{row['dataset']:<10} {row['baseline']:<12} {row['DM_stat']:>10.4f} {row['p_value']:>10.6f} {sig:>10} {row['interpretacja']}")

# Zlicz wyniki
n_total = len(summary_table)
n_significant = sum(1 for r in summary_table if r["istotne"])
n_hybrid_better = sum(1 for r in summary_table if r["istotne"] and r["DM_stat"] > 0)
n_baseline_better = sum(1 for r in summary_table if r["istotne"] and r["DM_stat"] < 0)
n_no_diff = sum(1 for r in summary_table if not r["istotne"])

print(f"\n  Laczna liczba porownan: {n_total}")
print(f"  Istotne statystycznie: {n_significant} ({n_significant/n_total*100:.1f}%)")
print(f"    - TCN+Mamdani lepszy: {n_hybrid_better}")
print(f"    - Baseline lepszy:    {n_baseline_better}")
print(f"  Brak istotnej roznicy: {n_no_diff}")

result_data = {
    "eksperyment": "Test istotnosci statystycznej Diebold-Mariano",
    "opis": "Porownanie TCN+Mamdani vs kazdy baseline na 4 zbiorach danych",
    "alpha": ALPHA,
    "H0": "Oba modele sa rownie dobre (brak roznic w dokladnosci prognoz)",
    "metoda_bledow": "e = actual - predicted, d = e1^2 - e2^2",
    "wyniki_po_datasecie": RESULTS,
    "tabela_podsumowania": summary_table,
    "statystyki_ogolne": {
        "liczba_porownan": n_total,
        "istotne_statystycznie": n_significant,
        "tcn_mamdani_lepszy": n_hybrid_better,
        "baseline_lepszy": n_baseline_better,
        "brak_roznicy": n_no_diff,
    },
}
save_results(result_data, os.path.join(os.path.dirname(__file__), "wyniki.json"))
print("\n=== EKSPERYMENT 14 ZAKONCZONE ===")
