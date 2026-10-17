# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 22: Wrazliwosc strategii z eksperymentu 17 na koszty transakcyjne.

Strategia identyczna jak w 17 (bez look-ahead): decyzja na dzien t na podstawie
ceny znanej actual[t-1]; long gdy predicted[t] > actual[t-1], w przeciwnym razie short.
Koszt c (w punktach bazowych) pobierany od kazdej transakcji jednostronnej:
  - otwarcie pierwszej pozycji: 1 transakcja,
  - zmiana kierunku (long <-> short): 2 transakcje (zamkniecie + otwarcie),
  - zamkniecie pozycji na koncu okresu: 1 transakcja.
Buy-and-hold placi koszt dwa razy (zakup na poczatku, sprzedaz na koncu).
Koszt progowy (break-even) = koszt, przy ktorym wynik netto strategii aktywnej
zrownuje sie z buy-and-hold netto (szukany bisekcja, tylko gdy aktywna strategia
bez kosztow jest lepsza od buy-and-hold).
Dane: prognozy z eksperymentow 01-05 (cztery glowne zbiory), bez trenowania modeli.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from utils import save_results
from config import DATASETS

OUT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(OUT)
PRED_DIRS = {
    "ARIMA": "01_arima_baseline",
    "LSTM": "02_lstm_baseline",
    "TCN": "03_tcn_baseline",
    "Mamdani": "04_mamdani_baseline",
    "TCN-Mamdani": "05_tcn_mamdani_hybrid",
}
MAIN = ["SP500", "WIG20", "EURUSD", "BTCUSD"]
COSTS_BPS = [0, 5, 10, 20]
INITIAL = 10000.0


def positions(actual, predicted):
    """Pozycja na dzien i (i>=1): +1 long, -1 short — jak w eksperymencie 17."""
    return np.array([1 if predicted[i] - actual[i - 1] > 0 else -1
                     for i in range(1, len(predicted))])


def simulate(actual, predicted, cost_bps):
    c = cost_bps / 10000.0
    pos = positions(actual, predicted)
    capital = INITIAL
    hist = [capital]
    n_tx = 0
    for k, i in enumerate(range(1, len(predicted))):
        if k == 0:
            capital *= (1 - c)
            n_tx += 1
        elif pos[k] != pos[k - 1]:
            capital *= (1 - c) ** 2
            n_tx += 2
        ret = (actual[i] - actual[i - 1]) / actual[i - 1]
        capital *= (1 + pos[k] * ret)
        hist.append(capital)
    capital *= (1 - c)
    n_tx += 1
    hist[-1] = capital
    hist = np.array(hist)
    peak = np.maximum.accumulate(hist)
    max_dd = float(np.max((peak - hist) / peak))
    total = capital / INITIAL - 1
    n_days = len(hist) - 1
    annual = (1 + total) ** (252 / max(n_days, 1)) - 1
    calmar = annual / max_dd if max_dd > 0 else float("inf")
    return {
        "total_return_pct": round(total * 100, 2),
        "calmar_ratio": round(calmar, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "liczba_transakcji": n_tx,
    }


def buy_hold(actual, cost_bps):
    c = cost_bps / 10000.0
    gross = actual[-1] / actual[0]
    return round(((1 - c) ** 2 * gross - 1) * 100, 2)


def break_even_bps(actual, predicted):
    """Koszt [bp], przy ktorym strategia aktywna netto = buy-and-hold netto."""
    def diff(bps):
        return simulate(actual, predicted, bps)["total_return_pct"] - buy_hold(actual, bps)
    if diff(0) <= 0:
        return None
    lo, hi = 0.0, 1000.0
    if diff(hi) > 0:
        return "powyzej 1000"
    for _ in range(60):
        mid = (lo + hi) / 2
        if diff(mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


RESULTS = {}
for ds in MAIN:
    ds_res = {}
    for model, d in PRED_DIRS.items():
        df = pd.read_csv(os.path.join(BASE, d, f"prognozy_{ds}.csv"))
        actual = df["Actual"].values
        predicted = df["Predicted"].values
        pos = positions(actual, predicted)
        flips = int(np.sum(pos[1:] != pos[:-1]))
        row = {
            "dni": int(len(pos)),
            "zmiany_kierunku": flips,
            "dni_long": int(np.sum(pos == 1)),
            "dni_short": int(np.sum(pos == -1)),
            "wyniki_wg_kosztu_bp": {str(b): simulate(actual, predicted, b) for b in COSTS_BPS},
            "buy_hold_wg_kosztu_bp": {str(b): buy_hold(actual, b) for b in COSTS_BPS},
            "koszt_progowy_bp_vs_buy_hold": break_even_bps(actual, predicted),
        }
        ds_res[model] = row
        r0 = row["wyniki_wg_kosztu_bp"]
        print(f"{DATASETS[ds]['name']:8s} {model:12s} zmian={flips:3d} "
              + " ".join(f"{b}bp={r0[str(b)]['total_return_pct']:7.2f}%" for b in COSTS_BPS)
              + f"  B&H0={row['buy_hold_wg_kosztu_bp']['0']}%  prog={row['koszt_progowy_bp_vs_buy_hold']}")
    RESULTS[ds] = ds_res

save_results({
    "eksperyment": "Wrazliwosc strategii (eksp. 17) na koszty transakcyjne",
    "koszty_bp": COSTS_BPS,
    "uwaga": "koszt na transakcje jednostronna; zmiana kierunku = 2 transakcje; "
             "otwarcie i zamkniecie okresu = po 1 transakcji; buy-and-hold = 2 transakcje",
    "wyniki": RESULTS,
}, os.path.join(OUT, "wyniki.json"))
print("\n=== EKSPERYMENT 22 ZAKONCZONY ===")
