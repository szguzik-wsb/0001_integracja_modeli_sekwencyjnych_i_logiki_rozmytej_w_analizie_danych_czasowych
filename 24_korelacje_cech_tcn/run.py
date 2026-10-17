# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 24: Czy wyjscia ekstraktora TCN (trend, zmiennosc, momentum) odpowiadaja
standardowym statystykom kroczacym?

Dla kazdego dnia testowego bierzemy cechy TCN zapisane w prognozach modelu hybrydowego
(05 dla S&P 500, WIG20, EUR/USD, BTC/USD; 20 dla DAX, Nikkei, Gold — te same przebiegi co
w Tabeli 3) oraz okno wejsciowe 30 cen poprzedzajacych dzien prognozy (to samo okno, ktore
widzial model). Z okna liczymy statystyki referencyjne (definicje ustalone przed liczeniem):
  - trend:      nachylenie prostej MNK dopasowanej do log-cen [% na sesje],
  - zmiennosc:  odchylenie standardowe dziennych log-stop zwrotu [%],
  - momentum:   10-sesyjna stopa zmiany ceny (rate of change) [%].
Raportujemy korelacje Pearsona i Spearmana (z p-value) kazdego wyjscia TCN z kazda statystyka,
zmiennosc samych wyjsc TCN (odchylenie, zakres) oraz odsetek dni z wyjsciem w poblizu
granicy dziedziny (|wartosc| >= 98% granicy).
Brak trenowania — wylacznie odczyt zapisanych cech i surowych cen.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy import stats
from utils import load_dataset, split_data, save_results
from config import DATASETS, LOOKBACK

OUT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(OUT)
SRC = {k: ("05_tcn_mamdani_hybrid", f"prognozy_{k}.csv") for k in ["SP500", "WIG20", "EURUSD", "BTCUSD"]}
SRC.update({k: ("20_all_7datasets", f"prognozy_hybrid_{k}.csv") for k in ["DAX", "NIKKEI", "GOLD"]})
FEATURES = {"TCN_trend": (-3, 3), "TCN_volatility": (0, 5), "TCN_momentum": (-3, 3)}


def window_stats(w):
    logp = np.log(w)
    x = np.arange(len(w))
    slope = np.polyfit(x, logp, 1)[0] * 100
    r = np.diff(logp) * 100
    vol = np.std(r, ddof=1)
    roc10 = (w[-1] / w[-11] - 1) * 100
    return slope, vol, roc10


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return {"pearson": None, "p_pearson": None, "spearman": None, "p_spearman": None}
    pr, pp = stats.pearsonr(a, b)
    sr, sp = stats.spearmanr(a, b)
    return {"pearson": round(float(pr), 3), "p_pearson": float(f"{pp:.3g}"),
            "spearman": round(float(sr), 3), "p_spearman": float(f"{sp:.3g}")}


RESULTS = {}
for ds_key, info in DATASETS.items():
    df = load_dataset(ds_key)
    _, _, test_df = split_data(df)
    prices = test_df["Close"].values
    d, f = SRC[ds_key]
    feat = pd.read_csv(os.path.join(BASE, d, f))
    n = len(prices) - LOOKBACK
    assert len(feat) == n, f"{ds_key}: {len(feat)} != {n}"
    # Kontrola zgodnosci okna: Actual w pliku = cena dnia prognozy
    assert np.allclose(feat["Actual"].values, prices[LOOKBACK:], rtol=1e-6)

    ref = np.array([window_stats(prices[i:i + LOOKBACK]) for i in range(n)])
    ref_names = ["nachylenie_logcen_pct", "odch_std_log_zwrotow_pct", "roc10_pct"]

    row = {"n_dni": int(n), "cechy_TCN": {}, "korelacje": {}}
    for fname, (lo, hi) in FEATURES.items():
        v = feat[fname].values.astype(float)
        bound = np.maximum(np.abs(lo), np.abs(hi))
        near = np.mean(np.abs(v) >= 0.98 * bound) if lo < 0 else np.mean(v >= 0.98 * hi)
        row["cechy_TCN"][fname] = {
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4),
            "srednia": round(float(v.mean()), 4), "odch_std": round(float(v.std(ddof=1)), 4),
            "odsetek_przy_granicy": round(float(near), 3),
        }
        row["korelacje"][fname] = {rn: corr(v, ref[:, j]) for j, rn in enumerate(ref_names)}
    row["statystyki_referencyjne"] = {
        rn: {"srednia": round(float(ref[:, j].mean()), 4), "odch_std": round(float(ref[:, j].std(ddof=1)), 4)}
        for j, rn in enumerate(ref_names)
    }
    RESULTS[ds_key] = row
    k = row["korelacje"]
    print(f"{info['name']:10s} r(trend,slope)={k['TCN_trend']['nachylenie_logcen_pct']['pearson']} "
          f"r(vol,std)={k['TCN_volatility']['odch_std_log_zwrotow_pct']['pearson']} "
          f"r(mom,roc10)={k['TCN_momentum']['roc10_pct']['pearson']} | sd: "
          + " ".join(f"{fn}={row['cechy_TCN'][fn]['odch_std']}" for fn in FEATURES))

save_results({
    "eksperyment": "Korelacje wyjsc ekstraktora TCN ze standardowymi statystykami kroczacymi (okno 30 sesji)",
    "definicje": {
        "nachylenie_logcen_pct": "nachylenie MNK log-cen w oknie 30 sesji, % na sesje",
        "odch_std_log_zwrotow_pct": "odchylenie standardowe 29 dziennych log-stop zwrotu w oknie, %",
        "roc10_pct": "zmiana ceny ostatniej obserwacji okna wzgledem obserwacji sprzed 10 sesji, %",
    },
    "zrodla_cech": {k: f"{v[0]}/{v[1]}" for k, v in SRC.items()},
    "wyniki": RESULTS,
}, os.path.join(OUT, "wyniki.json"))
print("\n=== EKSPERYMENT 24 ZAKONCZONY ===")
