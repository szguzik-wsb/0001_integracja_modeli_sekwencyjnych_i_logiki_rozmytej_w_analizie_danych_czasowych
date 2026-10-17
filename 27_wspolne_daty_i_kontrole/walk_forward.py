# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 27 (czesc: walidacja krocząca). Poprawione porownanie z eksperymentem 21.

W eksp. 21 ARIMA dostawala historie bez okresu walidacyjnego (po treningu od razu ceny testu)
i prognozowala caly test, a hybryda tylko dni test[LOOKBACK:] (pierwsze 30 dni testu sluzy jej
za okno wejsciowe). Tutaj, dla tych samych foldow co w 21:
  - ARIMA(5,1,0) szacowana w kazdym kroku na PELNEJ historii do dnia t-1
    (trening + walidacja + wczesniejsze dni testu),
  - prognoza naiwna = cena z dnia t-1,
obie na DOKLADNIE tych dniach, na ktorych liczono MAPE hybrydy w 21 (test[LOOKBACK:]).
MAPE hybrydy brane z 21/wyniki.json (bez ponownego trenowania).
Okna testowe kolejnych foldow nakladaja sie (krok < dlugosc testu) - liczone i zapisywane.
"""
import os, sys, json, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
OUT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(OUT)
sys.path.append(BASE)

import numpy as np
from multiprocessing import Pool
from utils import load_dataset, save_results
from config import DATASETS, LOOKBACK

N_FOLDS = 5
KEYS = ["SP500", "WIG20", "EURUSD", "BTCUSD"]


def folds(n):
    test_size, val_size, min_train = int(n * 0.15), int(n * 0.05), int(n * 0.40)
    step = (n - min_train - val_size - test_size) // (N_FOLDS - 1)
    out = []
    for f in range(N_FOLDS):
        train_end = min_train + f * step
        val_end = train_end + val_size
        test_end = min(val_end + test_size, n)
        out.append((f + 1, train_end, val_end, test_end))
    return out, step, test_size


def task(args):
    import warnings
    warnings.filterwarnings("ignore")
    from statsmodels.tsa.arima.model import ARIMA
    ds, fold, train_end, val_end, test_end = args
    prices = load_dataset(ds)["Close"].values
    idx = range(val_end + LOOKBACK, test_end)
    act = prices[list(idx)]
    naive = prices[[t - 1 for t in idx]]
    arima = []
    for t in idx:
        try:
            arima.append(ARIMA(prices[:t], order=(5, 1, 0)).fit().forecast(steps=1)[0])
        except Exception:
            arima.append(prices[t - 1])
    m = lambda f: float(np.mean(np.abs((act - np.asarray(f)) / act)) * 100)
    return ds, fold, {"n": len(act), "MAPE_ARIMA_pelna_historia": round(m(arima), 4),
                      "MAPE_naiwna": round(m(naive), 4)}


if __name__ == "__main__":
    w21 = json.load(open(os.path.join(BASE, "21_walk_forward", "wyniki.json"), encoding="utf-8"))
    jobs, meta = [], {}
    for ds in KEYS:
        df = load_dataset(ds)
        fl, step, test_size = folds(len(df))
        meta[ds] = {"n": len(df), "krok": step, "dlugosc_testu": test_size,
                    "nakladanie_kolejnych_testow_dni": max(0, test_size - step)}
        for f, tr, ve, te in fl:
            jobs.append((ds, f, tr, ve, te))
            meta[ds][f] = {"test_od": str(df["Date"].iloc[ve + LOOKBACK])[:10], "test_do": str(df["Date"].iloc[te - 1])[:10]}
    with Pool(min(len(jobs), os.cpu_count() or 4)) as pool:
        res = pool.map(task, jobs)
    wyn = {}
    for ds, f, r in sorted(res, key=lambda x: (KEYS.index(x[0]), x[1])):
        old = next(x for x in w21[ds]["folds"] if x["fold"] == f)
        r.update({"MAPE_hybryda_eksp21": old["hybrid"]["MAPE"], "MAPE_ARIMA_eksp21_(inna_probka)": old["arima"]["MAPE"],
                  "okres_testu": f"{meta[ds][f]['test_od']} - {meta[ds][f]['test_do']}"})
        wyn.setdefault(ds, {"foldy": []})["foldy"].append(dict(fold=f, **r))
        print(f"{DATASETS[ds]['name']:8s} fold {f}: n={r['n']} hybryda={r['MAPE_hybryda_eksp21']} "
              f"ARIMA={r['MAPE_ARIMA_pelna_historia']} naiwna={r['MAPE_naiwna']} (ARIMA w 21: {r['MAPE_ARIMA_eksp21_(inna_probka)']})", flush=True)
    for ds in KEYS:
        F = wyn[ds]["foldy"]
        h = np.array([x["MAPE_hybryda_eksp21"] for x in F])
        a = np.array([x["MAPE_ARIMA_pelna_historia"] for x in F])
        nv = np.array([x["MAPE_naiwna"] for x in F])
        wyn[ds]["podsumowanie"] = {
            "hybryda_srednia": round(h.mean(), 4), "hybryda_odch": round(h.std(), 4),
            "ARIMA_srednia": round(a.mean(), 4), "ARIMA_odch": round(a.std(), 4),
            "naiwna_srednia": round(nv.mean(), 4), "naiwna_odch": round(nv.std(), 4),
            "hybryda_lepsza_od_ARIMA_foldy": int(np.sum(h < a)), "hybryda_lepsza_od_naiwnej_foldy": int(np.sum(h < nv)),
            "ARIMA_lepsza_od_naiwnej_foldy": int(np.sum(a < nv)),
            "krok_dni": meta[ds]["krok"], "dlugosc_testu_dni": meta[ds]["dlugosc_testu"],
            "nakladanie_kolejnych_testow_dni": meta[ds]["nakladanie_kolejnych_testow_dni"]}
        print(DATASETS[ds]["name"], wyn[ds]["podsumowanie"], flush=True)
    save_results({"eksperyment": "27: walidacja kroczaca - ARIMA z pelna historia i prognoza naiwna na probce hybrydy z eksp. 21",
                  "wyniki": wyn}, os.path.join(OUT, "wyniki_walk_forward.json"))
