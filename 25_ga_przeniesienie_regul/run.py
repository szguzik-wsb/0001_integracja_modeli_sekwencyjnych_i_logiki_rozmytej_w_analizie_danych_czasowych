# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 25: Przenoszalnosc bazy regul zoptymalizowanej algorytmem genetycznym na S&P 500
(eksperyment 19) na pozostale zbiory danych.

Ekstraktor TCN NIE jest trenowany ponownie: uzywamy cech TCN zapisanych w prognozach modelu
hybrydowego (05 dla S&P 500, WIG20, EUR/USD, BTC/USD; 20 dla DAX, Nikkei, Gold — te same
przebiegi co w Tabeli 3). Podmieniana jest wylacznie warstwa reguł:
  - "reczne": 15 regul i parametry MF z eksperymentu 05 (kontrola: odtworzenie MAPE z Tabeli 3),
  - "GA_SP500": 10 regul i parametry MF wybrane przez GA na zbiorze walidacyjnym S&P 500 (19).
Dodatkowo: profil aktywacji regul (prog 0,1, jak w eksperymencie 12) dla obu baz na kazdym
zbiorze oraz odsetek prognoz wzrostu (diagnoza degeneracji kierunkowej).
To jednoczesnie praktyczny przyklad interwencji w warstwie regul bez ponownego trenowania sieci.
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from utils import prepare_data, compute_all_metrics, inverse_transform, diebold_mariano, save_results
from config import DATASETS

OUT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(OUT)
SRC = {k: ("05_tcn_mamdani_hybrid", f"prognozy_{k}.csv") for k in ["SP500", "WIG20", "EURUSD", "BTCUSD"]}
SRC.update({k: ("20_all_7datasets", f"prognozy_hybrid_{k}.csv") for k in ["DAX", "NIKKEI", "GOLD"]})

T5 = ["duzy_spadek", "spadek", "neutralny", "wzrost", "duzy_wzrost"]
T3 = ["niska", "srednia", "wysoka"]
# Parametry reczne z eksperymentu 05 (centrum, szerokosc)
MANUAL = {
    "trend": dict(zip(T5, [(-2.5, 0.6), (-1, 0.4), (0, 0.35), (1, 0.4), (2.5, 0.6)])),
    "zmiennosc": dict(zip(T3, [(0, 0.5), (2, 0.6), (4, 0.7)])),
    "momentum": dict(zip(T5, [(-2.5, 0.6), (-1, 0.4), (0, 0.35), (1, 0.4), (2.5, 0.6)])),
    "prognoza": dict(zip(T5, [(-2.5, 0.6), (-1, 0.4), (0, 0.35), (1, 0.4), (2.5, 0.6)])),
}
# (antecedent1, antecedent2, consequent) — kolejnosc R1..R15 jak w 05/19
RULES = [
    (("trend", "duzy_wzrost"), ("momentum", "duzy_wzrost"), "duzy_wzrost"),
    (("trend", "wzrost"), ("momentum", "wzrost"), "wzrost"),
    (("trend", "spadek"), ("momentum", "spadek"), "spadek"),
    (("trend", "duzy_spadek"), ("momentum", "duzy_spadek"), "duzy_spadek"),
    (("trend", "wzrost"), ("momentum", "spadek"), "neutralny"),
    (("trend", "spadek"), ("momentum", "wzrost"), "neutralny"),
    (("trend", "neutralny"), ("momentum", "neutralny"), "neutralny"),
    (("trend", "neutralny"), ("momentum", "wzrost"), "wzrost"),
    (("trend", "neutralny"), ("momentum", "spadek"), "spadek"),
    (("trend", "duzy_wzrost"), ("zmiennosc", "wysoka"), "neutralny"),
    (("trend", "duzy_spadek"), ("zmiennosc", "wysoka"), "neutralny"),
    (("trend", "wzrost"), ("zmiennosc", "niska"), "duzy_wzrost"),
    (("trend", "spadek"), ("zmiennosc", "niska"), "duzy_spadek"),
    (("momentum", "duzy_wzrost"), ("zmiennosc", "srednia"), "wzrost"),
    (("momentum", "duzy_spadek"), ("zmiennosc", "srednia"), "spadek"),
]


def load_ga_params():
    with open(os.path.join(BASE, "19_ga_reguly_optymalizacja", "wyniki.json"), encoding="utf-8") as f:
        res19 = json.load(f)
    ga = res19["ga_optymalizowane"]
    p = res19["optimized_mf_params"]
    params = {}
    for var, terms in [("trend", T5), ("zmiennosc", T3), ("momentum", T5), ("prognoza", T5)]:
        params[var] = {t: (p[f"{var}_{t}"]["center"], max(p[f"{var}_{t}"]["width"], 0.1)) for t in terms}
    return params, [r - 1 for r in ga["reguly_aktywne"]]


def build(params, active_idx):
    v = {
        "trend": ctrl.Antecedent(np.linspace(-3, 3, 1000), "trend"),
        "zmiennosc": ctrl.Antecedent(np.linspace(0, 5, 1000), "zmiennosc"),
        "momentum": ctrl.Antecedent(np.linspace(-3, 3, 1000), "momentum"),
        "prognoza": ctrl.Consequent(np.linspace(-5, 5, 1000), "prognoza"),
    }
    for var, terms in params.items():
        for t, (c, w) in terms.items():
            v[var][t] = fuzz.gaussmf(v[var].universe, c, w)
    rules = [ctrl.Rule(v[a1[0]][a1[1]] & v[a2[0]][a2[1]], v["prognoza"][c])
             for k, (a1, a2, c) in enumerate(RULES) if k in active_idx]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))


def clip(f):
    return np.clip(f[0], -2.9, 2.9), np.clip(f[1], 0.01, 4.9), np.clip(f[2], -2.9, 2.9)


def predict(sim, f):
    t, z, m = clip(f)
    sim.input["trend"], sim.input["zmiennosc"], sim.input["momentum"] = t, z, m
    try:
        sim.compute()
        return sim.output["prognoza"]
    except Exception:
        return 0.0


def firing(params, f):
    """Stopnie aktywacji 15 regul (t-norma min) dla jednego wektora cech."""
    vals = dict(zip(["trend", "zmiennosc", "momentum"], clip(f)))
    def mu(var, term):
        c, w = params[var][term]
        return float(np.exp(-((vals[var] - c) ** 2) / (2 * w ** 2)))
    return np.array([min(mu(*a1), mu(*a2)) for a1, a2, _ in RULES])


GA_PARAMS, GA_ACTIVE = load_ga_params()
SYSTEMS = {"reczne": (MANUAL, list(range(15))), "GA_SP500": (GA_PARAMS, GA_ACTIVE)}

RESULTS = {}
for ds_key, info in DATASETS.items():
    data = prepare_data(ds_key)
    d, fname = SRC[ds_key]
    df = pd.read_csv(os.path.join(BASE, d, fname))
    feats = df[["TCN_trend", "TCN_volatility", "TCN_momentum"]].values.astype(float)
    assert len(feats) == len(data["X_test"]), ds_key
    last = data["X_test"][:, -1]
    scaler = data["scaler"]
    actual = inverse_transform(scaler, data["y_test"])
    prev = inverse_transform(scaler, last)
    row = {}
    preds_all = {}
    for name, (params, active) in SYSTEMS.items():
        sim = build(params, active)
        change = np.array([predict(sim, f) for f in feats])
        pred = inverse_transform(scaler, last * (1 + change / 100))
        preds_all[name] = pred
        fire = np.array([firing(params, f) for f in feats])[:, active]
        freq = (fire > 0.1).mean(axis=0)
        row[name] = {
            **compute_all_metrics(actual, pred),
            "odsetek_prognoz_wzrostu": round(float(np.mean(pred > prev)), 3),
            "aktywne_reguly": [k + 1 for k in active],
            "czestosc_aktywacji_powyzej_0_1": {f"R{k + 1}": round(float(q), 3) for k, q in zip(active, freq)},
            "srednia_liczba_aktywnych_regul": round(float((fire > 0.1).sum(axis=1).mean()), 2),
        }
    # Kontrola: reczne odtwarza prognozy z pliku zrodlowego
    row["kontrola_max_roznica_vs_plik"] = round(float(np.max(np.abs(preds_all["reczne"] - df["Predicted"].values)
                                                         / df["Actual"].values) * 100), 6)
    e_m = actual - preds_all["reczne"]
    e_g = actual - preds_all["GA_SP500"]
    stat, p = diebold_mariano(e_m, e_g)
    row["DM_GA_vs_reczne"] = {"DM": round(float(stat), 3), "p": float(f"{p:.3g}"), "n": int(len(actual)),
                              "konwencja": "dodatnia = baza GA ma mniejszy blad"}
    RESULTS[ds_key] = row
    print(f"{info['name']:10s} MAPE reczne={row['reczne']['MAPE']:.4f} GA={row['GA_SP500']['MAPE']:.4f} "
          f"DM={row['DM_GA_vs_reczne']['DM']} p={row['DM_GA_vs_reczne']['p']} | wzrost% r={row['reczne']['odsetek_prognoz_wzrostu']} "
          f"ga={row['GA_SP500']['odsetek_prognoz_wzrostu']} | kontrola={row['kontrola_max_roznica_vs_plik']}%")

save_results({
    "eksperyment": "Przenoszalnosc bazy regul GA (dostrojonej na S&P 500) na 7 zbiorow bez ponownego trenowania TCN",
    "zrodla_cech": {k: f"{v[0]}/{v[1]}" for k, v in SRC.items()},
    "prog_aktywacji": 0.1,
    "wyniki": RESULTS,
}, os.path.join(OUT, "wyniki.json"))
print("\n=== EKSPERYMENT 25 ZAKONCZONY ===")
