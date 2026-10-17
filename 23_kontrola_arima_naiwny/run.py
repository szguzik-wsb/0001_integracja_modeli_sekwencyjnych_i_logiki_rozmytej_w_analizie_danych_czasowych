# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 23: Kontrola asymetrii procedur ewaluacji (ARIMA walk-forward vs DL trenowane raz).

Warianty porownawcze na tych samych dniach testowych co eksperyment 01:
  (a) ARIMA(5,1,0) z re-estymacja w kazdym kroku — prognozy z eksperymentu 01 (bez zmian),
  (b) ARIMA(5,1,0) ze STALYMI parametrami: estymacja jeden raz na train+val, nastepnie
      prognozy jednokrokowe na zbiorze testowym z filtrem Kalmana przy tych samych
      parametrach (informacja do t-1, bez ponownej estymacji) — ten sam rezim
      informacyjny co modele glebokie trenowane raz,
  (c) prognoza naiwna (random walk): P_hat[t] = P[t-1].
Dla kazdego wariantu: MAPE/RMSE/MAE na 7 zbiorach; test Diebolda-Mariano wzgledem
TCN-Mamdani (prognozy wyrownane po dacie; 4 glowne zbiory z 05, DAX/Nikkei/Gold z 20)
oraz wzgledem siebie nawzajem.
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from utils import load_dataset, split_data, compute_all_metrics, diebold_mariano, save_results
from config import DATASETS

OUT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(OUT)
HYBRID_SRC = {k: ("05_tcn_mamdani_hybrid", f"prognozy_{k}.csv") for k in ["SP500", "WIG20", "EURUSD", "BTCUSD"]}
HYBRID_SRC.update({k: ("20_all_7datasets", f"prognozy_hybrid_{k}.csv") for k in ["DAX", "NIKKEI", "GOLD"]})
ARIMA_SRC = {k: ("01_arima_baseline", f"prognozy_{k}.csv") for k in DATASETS}


def norm_dates(s):
    return pd.to_datetime(s).dt.normalize()


def dm_pair(df_a, df_b):
    """DM dla dwoch ramek (Date, Actual, Predicted) po wspolnych datach.
    Konwencja jak w Tabeli 4 (eksp. 14): wartosc dodatnia = model a ma mniejszy blad (a lepszy)."""
    m = pd.merge(df_a, df_b, on="Date", suffixes=("_a", "_b"))
    assert np.allclose(m["Actual_a"], m["Actual_b"], rtol=1e-9), "rozne wartosci rzeczywiste"
    e_a = m["Actual_a"] - m["Predicted_a"]
    e_b = m["Actual_b"] - m["Predicted_b"]
    stat, p = diebold_mariano(e_b, e_a)
    return {"DM": round(float(stat), 4), "p": float(f"{p:.3g}"), "n": int(len(m))}


RESULTS = {}
for ds_key, info in DATASETS.items():
    t0 = time.time()
    df = load_dataset(ds_key)
    train_df, val_df, test_df = split_data(df)
    hist = np.concatenate([train_df["Close"].values, val_df["Close"].values])
    test_vals = test_df["Close"].values
    full = np.concatenate([hist, test_vals])

    # (b) ARIMA ze stalymi parametrami
    res_hist = ARIMA(hist, order=(5, 1, 0)).fit()
    res_full = ARIMA(full, order=(5, 1, 0)).filter(res_hist.params)
    pred_fixed = res_full.predict(start=len(hist), end=len(full) - 1)

    # (c) prognoza naiwna
    pred_naive = full[len(hist) - 1:len(full) - 1]

    dates = norm_dates(test_df["Date"])
    fixed_df = pd.DataFrame({"Date": dates.values, "Actual": test_vals, "Predicted": pred_fixed})
    naive_df = pd.DataFrame({"Date": dates.values, "Actual": test_vals, "Predicted": pred_naive})
    fixed_df.to_csv(os.path.join(OUT, f"prognozy_arima_stale_{ds_key}.csv"), index=False)
    naive_df.to_csv(os.path.join(OUT, f"prognozy_naiwna_{ds_key}.csv"), index=False)

    a_dir, a_file = ARIMA_SRC[ds_key]
    reest_df = pd.read_csv(os.path.join(BASE, a_dir, a_file))
    reest_df["Date"] = norm_dates(reest_df["Date"])
    h_dir, h_file = HYBRID_SRC[ds_key]
    hyb_df = pd.read_csv(os.path.join(BASE, h_dir, h_file))[["Date", "Actual", "Predicted"]]
    hyb_df["Date"] = norm_dates(hyb_df["Date"])

    row = {
        "parametry_ARIMA_stale": {k: round(float(v), 6) for k, v in zip(res_hist.param_names, res_hist.params)},
        "ARIMA_reestymowana": compute_all_metrics(reest_df["Actual"], reest_df["Predicted"]),
        "ARIMA_stale_parametry": compute_all_metrics(test_vals, pred_fixed),
        "Naiwna": compute_all_metrics(test_vals, pred_naive),
        "DM": {
            "TCN-Mamdani_vs_ARIMA_reestymowana": dm_pair(hyb_df, reest_df),
            "TCN-Mamdani_vs_ARIMA_stale_parametry": dm_pair(hyb_df, fixed_df),
            "TCN-Mamdani_vs_Naiwna": dm_pair(hyb_df, naive_df),
            "ARIMA_stale_vs_ARIMA_reestymowana": dm_pair(fixed_df, reest_df),
            "ARIMA_reestymowana_vs_Naiwna": dm_pair(reest_df, naive_df),
            "ARIMA_stale_vs_Naiwna": dm_pair(fixed_df, naive_df),
        },
        "czas_s": round(time.time() - t0, 2),
    }
    RESULTS[ds_key] = row
    print(f"{info['name']:10s} MAPE reest={row['ARIMA_reestymowana']['MAPE']:.4f} "
          f"stale={row['ARIMA_stale_parametry']['MAPE']:.4f} naiwna={row['Naiwna']['MAPE']:.4f} | "
          + " ".join(f"{k}: {v['DM']} (p={v['p']}, n={v['n']})" for k, v in row["DM"].items()))

save_results({
    "eksperyment": "Kontrola asymetrii: ARIMA re-estymowana vs ARIMA ze stalymi parametrami vs prognoza naiwna",
    "konwencja_DM": "statystyka dla pary 'X_vs_Y': dodatnia = X ma mniejszy blad kwadratowy (X lepszy); "
                    "liczone jako DM(e_Y, e_X) po wspolnych datach",
    "wyniki": RESULTS,
}, os.path.join(OUT, "wyniki.json"))
print("\n=== EKSPERYMENT 23 ZAKONCZONY ===")
