# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 27: kontrole na wspolnych datach (bez trenowania modeli).

Wszystkie liczby licza sie z zapisanych prognoz (01-05, 20, 23) i danych zrodlowych:
  A. MAPE wszystkich modeli na WSPOLNYCH datach (te same dni dla kazdego modelu) oraz test
     Diebolda-Mariano w dwoch wariantach: jak w eksp. 14 (bez korekty autokorelacji) i z
     wariancja Neweya-Westa (jadro Bartletta, L = floor(4 (n/100)^(2/9))) jako test odpornosci.
  B. Strategia z eksp. 17/22 na wspolnych datach: koszty 0/5/10/20 pb, koszt progowy wzgledem
     kup-i-trzymaj i wzgledem zera, liczba odwrocen pozycji, Calmar (BTC/USD: 365 dni w roku,
     pozostale: 252), krzywe kapitalu do rysunku.
  C. Rezimy zmiennosci ex ante: odchylenie std. dziennych log-zwrotow z 20 dni znane w dniu t-1;
     rezim wysoki = gorny kwartyl tej zmiennosci na wspolnych datach testu.
  E. Wariant wzoru w skali cen: ta sama zmiana z Mamdaniego zastosowana do ostatniej CENY zamiast
     ostatniej wartosci znormalizowanej (cechy TCN z pliku prognoz, bez trenowania).
  F. Odsetek dni testu z cena powyzej maksimum / ponizej minimum zbioru treningowego.
  G. Korelacje wyjsc TCN z poziomem ceny i miedzy soba (p nominalne: okna sie nakladaja).
  H. Sila odpalenia regul (srednia i odsetek dni z sila > 0,1) na kazdym zbiorze.
  I. Prognoza naiwna na probkach eksperymentu 11 (rozne okna wejsciowe, S&P 500).
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
OUT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(OUT)
sys.path.append(BASE)

import numpy as np
import pandas as pd
from scipy import stats
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from utils import load_dataset, prepare_data, diebold_mariano, save_results
from config import DATASETS, LOOKBACK

MAIN = ["SP500", "WIG20", "EURUSD", "BTCUSD"]
EXTRA = ["DAX", "NIKKEI", "GOLD"]
ALL7 = MAIN + EXTRA
HYB = {k: ("05_tcn_mamdani_hybrid", f"prognozy_{k}.csv") for k in MAIN}
HYB.update({k: ("20_all_7datasets", f"prognozy_hybrid_{k}.csv") for k in EXTRA})
OTHER = {"ARIMA": "01_arima_baseline", "LSTM": "02_lstm_baseline",
         "TCN": "03_tcn_baseline", "Mamdani": "04_mamdani_baseline"}
STRAT_MODELS = ["ARIMA", "LSTM", "TCN", "Mamdani", "TCN-Mamdani"]
COSTS_BPS = [0, 5, 10, 20]
INITIAL = 10000.0
R = lambda x, k=4: None if x is None else round(float(x), k)


def load_pred(folder, fname):
    df = pd.read_csv(os.path.join(BASE, folder, fname))
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df.drop_duplicates("Date").set_index("Date")


def all_preds(ds):
    p = {m: load_pred(f, f"prognozy_{ds}.csv") for m, f in OTHER.items()}
    p["TCN-Mamdani"] = load_pred(*HYB[ds])
    p["ARIMA_stale"] = load_pred("23_kontrola_arima_naiwny", f"prognozy_arima_stale_{ds}.csv")
    p["Naiwna"] = load_pred("23_kontrola_arima_naiwny", f"prognozy_naiwna_{ds}.csv")
    common = None
    for df in p.values():
        common = df.index if common is None else common.intersection(df.index)
    common = common.sort_values()
    act = p["TCN-Mamdani"].loc[common, "Actual"].values
    zg = max(float(np.max(np.abs(df.loc[common, "Actual"].values - act) / act)) for df in p.values())
    return {m: df.loc[common] for m, df in p.items()}, common, zg


def mape(a, f):
    return float(np.mean(np.abs((a - f) / a)) * 100)


def dm_nw(e_a, e_b):
    """DM z wariancja Neweya-Westa. d = e_b^2 - e_a^2; wartosc dodatnia => model a lepszy."""
    d = e_b ** 2 - e_a ** 2
    n = len(d)
    L = int(np.floor(4 * (n / 100) ** (2 / 9)))
    dc = d - d.mean()
    s = dc @ dc / n
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(s / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    rho1 = float(np.corrcoef(d[1:], d[:-1])[0, 1])
    return stat, p, L, rho1


def dm_both(df_a, df_b):
    e_a = (df_a["Actual"] - df_a["Predicted"]).values
    e_b = (df_b["Actual"] - df_b["Predicted"]).values
    s, p = diebold_mariano(e_b, e_a, h=1)
    s2, p2, L, rho1 = dm_nw(e_a, e_b)
    return {"DM": R(s, 3), "p": float(f"{p:.3g}"), "DM_NW": R(s2, 3), "p_NW": float(f"{p2:.3g}"),
            "L_NW": L, "rho1_d": R(rho1, 3), "n": len(e_a)}


# --------------------------------------------------------------------------- strategia
def positions(actual, predicted):
    return np.array([1 if predicted[i] - actual[i - 1] > 0 else -1 for i in range(1, len(predicted))])


def simulate(actual, predicted, cost_bps, ann):
    c = cost_bps / 10000.0
    pos = positions(actual, predicted)
    capital = INITIAL
    hist = [capital]
    for k, i in enumerate(range(1, len(predicted))):
        if k == 0:
            capital *= (1 - c)
        elif pos[k] != pos[k - 1]:
            capital *= (1 - c) ** 2
        capital *= (1 + pos[k] * (actual[i] - actual[i - 1]) / actual[i - 1])
        hist.append(capital)
    capital *= (1 - c)
    hist[-1] = capital
    return stats_from_curve(np.array(hist), ann)


def stats_from_curve(hist, ann):
    peak = np.maximum.accumulate(hist)
    max_dd = float(np.max((peak - hist) / peak))
    total = hist[-1] / INITIAL - 1          # wzgledem kapitalu poczatkowego (przed kosztem wejscia)
    n_days = len(hist) - 1
    annual = (1 + total) ** (ann / max(n_days, 1)) - 1
    return {"total_return_pct": round(total * 100, 2), "annual_return_pct": round(annual * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "calmar_ratio": round(annual / max_dd, 3) if max_dd > 0 else None, "curve": hist}


def buy_hold(actual, cost_bps, ann):
    c = cost_bps / 10000.0
    hist = INITIAL * (1 - c) * actual / actual[0]
    hist[-1] *= (1 - c)
    return stats_from_curve(hist, ann)


def bisect(f, lo=0.0, hi=1000.0):
    if f(lo) <= 0:
        return None
    if f(hi) > 0:
        return "powyzej 1000"
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return round((lo + hi) / 2, 2)


# --------------------------------------------------------------------------- Mamdani (z 05/run.py)
GAUSS = {"duzy_spadek": (-2.5, 0.6), "spadek": (-1, 0.4), "neutralny": (0, 0.35),
         "wzrost": (1, 0.4), "duzy_wzrost": (2.5, 0.6)}
VOLMF = {"niska": (0, 0.5), "srednia": (2, 0.6), "wysoka": (4, 0.7)}
RULES = [("t", "duzy_wzrost", "m", "duzy_wzrost"), ("t", "wzrost", "m", "wzrost"),
         ("t", "spadek", "m", "spadek"), ("t", "duzy_spadek", "m", "duzy_spadek"),
         ("t", "wzrost", "m", "spadek"), ("t", "spadek", "m", "wzrost"),
         ("t", "neutralny", "m", "neutralny"), ("t", "neutralny", "m", "wzrost"),
         ("t", "neutralny", "m", "spadek"), ("t", "duzy_wzrost", "v", "wysoka"),
         ("t", "duzy_spadek", "v", "wysoka"), ("t", "wzrost", "v", "niska"),
         ("t", "spadek", "v", "niska"), ("m", "duzy_wzrost", "v", "srednia"),
         ("m", "duzy_spadek", "v", "srednia")]
CONSEQ = ["duzy_wzrost", "wzrost", "spadek", "duzy_spadek", "neutralny", "neutralny", "neutralny",
          "wzrost", "spadek", "neutralny", "neutralny", "duzy_wzrost", "duzy_spadek", "wzrost", "spadek"]


def build_mamdani():
    trend = ctrl.Antecedent(np.linspace(-3, 3, 1000), "trend")
    zmiennosc = ctrl.Antecedent(np.linspace(0, 5, 1000), "zmiennosc")
    momentum = ctrl.Antecedent(np.linspace(-3, 3, 1000), "momentum")
    prognoza = ctrl.Consequent(np.linspace(-5, 5, 1000), "prognoza")
    for var in [trend, momentum, prognoza]:
        for name, (m, s) in GAUSS.items():
            var[name] = fuzz.gaussmf(var.universe, m, s)
    for name, (m, s) in VOLMF.items():
        zmiennosc[name] = fuzz.gaussmf(zmiennosc.universe, m, s)
    V = {"t": trend, "m": momentum, "v": zmiennosc}
    rules = [ctrl.Rule(V[a][la] & V[b][lb], prognoza[c]) for (a, la, b, lb), c in zip(RULES, CONSEQ)]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))


def mamdani_predict(sim, f):
    sim.input["trend"] = np.clip(f[0], -2.9, 2.9)
    sim.input["zmiennosc"] = np.clip(f[1], 0.01, 4.9)
    sim.input["momentum"] = np.clip(f[2], -2.9, 2.9)
    try:
        sim.compute()
        return sim.output["prognoza"]
    except Exception:
        return 0.0


def rule_strengths(F):
    x = {"t": np.clip(F[:, 0], -2.9, 2.9), "v": np.clip(F[:, 1], 0.01, 4.9), "m": np.clip(F[:, 2], -2.9, 2.9)}
    g = lambda v, ms: np.exp(-((v - ms[0]) ** 2) / (2 * ms[1] ** 2))
    mf = lambda var, lab: g(x[var], VOLMF[lab] if var == "v" else GAUSS[lab])
    return np.array([np.minimum(mf(a, la), mf(b, lb)) for a, la, b, lb in RULES]).T


# =========================================================================== MAIN
WYN = {"A_mape_i_DM_wspolne_daty": {}, "B_strategia_wspolne_daty": {}, "C_rezimy_ex_ante": {},
       "E_wariant_skali_cen": {}, "F_poza_zakresem_treningu": {}, "G_korelacje": {},
       "H_reguly": {}, "I_lookback_naiwna_SP500": {}}
CURVES = {}

for ds in ALL7:
    name = DATASETS[ds]["name"]
    P, common, zg = all_preds(ds)
    H = P["TCN-Mamdani"]
    act = H["Actual"].values

    # ---------------- A
    a = {"n_wspolnych_dat": len(common), "od": str(common[0].date()), "do": str(common[-1].date()),
         "max_wzgl_roznica_Actual_miedzy_plikami": float(f"{zg:.2e}"),
         "MAPE": {m: R(mape(df["Actual"].values, df["Predicted"].values)) for m, df in P.items()},
         "DM_hybryda_vs": {m: dm_both(H, P[m]) for m in ["ARIMA", "ARIMA_stale", "Naiwna", "LSTM", "TCN", "Mamdani"]},
         "DM_ARIMA_vs_Naiwna": dm_both(P["ARIMA"], P["Naiwna"]),
         "DM_ARIMA_stale_vs_Naiwna": dm_both(P["ARIMA_stale"], P["Naiwna"]),
         "DM_ARIMA_stale_vs_ARIMA": dm_both(P["ARIMA_stale"], P["ARIMA"])}
    WYN["A_mape_i_DM_wspolne_daty"][ds] = a
    print(f"\n{name}: n={len(common)} {a['od']}..{a['do']} MAPE " +
          " ".join(f"{m}={v}" for m, v in a["MAPE"].items()))
    for m, v in a["DM_hybryda_vs"].items():
        print(f"   DM hyb vs {m:12s} {v['DM']:8.3f} (p={v['p']:.3g})  NW {v['DM_NW']:8.3f} (p={v['p_NW']:.3g}) rho1={v['rho1_d']}")

    # ---------------- B (cztery glowne zbiory)
    if ds in MAIN:
        ann = 365 if ds == "BTCUSD" else 252
        b = {"dni_roku_do_annualizacji": ann, "modele": {}}
        bh = {str(c): buy_hold(act, c, ann) for c in COSTS_BPS}
        b["buy_hold"] = {k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in bh.items()}
        curves = {"Date": [d.date().isoformat() for d in common], "Buy-and-hold": bh["0"]["curve"]}
        for m in STRAT_MODELS:
            pr = P[m]["Predicted"].values
            pos = positions(act, pr)
            res = {str(c): simulate(act, pr, c, ann) for c in COSTS_BPS}
            curves[m] = res["0"]["curve"]
            b["modele"][m] = {
                "zmiany_kierunku": int(np.sum(pos[1:] != pos[:-1])),
                "dni_long": int(np.sum(pos == 1)), "dni_short": int(np.sum(pos == -1)),
                "wyniki_wg_kosztu_bp": {k: {kk: vv for kk, vv in v.items() if kk != "curve"} for k, v in res.items()},
                "koszt_progowy_bp_vs_buy_hold": bisect(
                    lambda c: simulate(act, pr, c, ann)["total_return_pct"] - buy_hold(act, c, ann)["total_return_pct"]),
                "koszt_progowy_bp_zysk_zero": bisect(lambda c: simulate(act, pr, c, ann)["total_return_pct"]),
            }
            r0 = b["modele"][m]
            print(f"   strat {m:12s} zmian={r0['zmiany_kierunku']:3d} " +
                  " ".join(f"{c}bp={r0['wyniki_wg_kosztu_bp'][str(c)]['total_return_pct']:7.2f}%" for c in COSTS_BPS) +
                  f" | B&H0={b['buy_hold']['0']['total_return_pct']}% prog_BH={r0['koszt_progowy_bp_vs_buy_hold']}"
                  f" prog_0={r0['koszt_progowy_bp_zysk_zero']} Calmar0={r0['wyniki_wg_kosztu_bp']['0']['calmar_ratio']}")
        WYN["B_strategia_wspolne_daty"][ds] = b
        # K: trafnosc kierunkowa na wspolnych datach (jak w eksp. 15: kierunek prognozy wzgledem ceny z t-1)
        WYN.setdefault("K_trafnosc_kierunkowa_wspolne_daty", {})[ds] = {}
        ad = act[1:] - act[:-1]
        for m in STRAT_MODELS:
            pdir = P[m]["Predicted"].values[1:] - act[:-1]
            ok = np.sign(ad) == np.sign(pdir)
            WYN["K_trafnosc_kierunkowa_wspolne_daty"][ds][m] = {
                "DA": R(100 * ok.mean(), 2), "DA_up": R(100 * ok[ad > 0].mean(), 2), "DA_down": R(100 * ok[ad <= 0].mean(), 2),
                "n": int(len(ad)), "odsetek_prognoz_wzrostu": R(100 * np.mean(pdir > 0), 1)}
        pd.DataFrame(curves).to_csv(os.path.join(OUT, f"krzywe_kapitalu_{ds}.csv"), index=False)

    # ---------------- C (rezimy ex ante, cztery glowne zbiory)
    full = load_dataset(ds)
    full["Date"] = pd.to_datetime(full["Date"]).dt.normalize()
    full = full.drop_duplicates("Date").set_index("Date")
    lr = np.log(full["Close"]).diff()
    vol_prev = lr.rolling(20).std().shift(1)          # znane na zamknieciu dnia t-1
    v = vol_prev.reindex(common).values
    if ds in MAIN:
        thr = float(np.nanquantile(v, 0.75))
        hi = v >= thr
        c_res = {"prog_zmiennosci_20d_pct": R(thr * 100, 4), "dni_wysoka": int(hi.sum()), "dni_pozostale": int((~hi).sum()),
                 "MAPE_wysoka": {m: R(mape(df["Actual"].values[hi], df["Predicted"].values[hi])) for m, df in P.items()},
                 "MAPE_pozostale": {m: R(mape(df["Actual"].values[~hi], df["Predicted"].values[~hi])) for m, df in P.items()},
                 "DM_wysoka_hybryda_vs": {m: dm_both(H[hi], P[m][hi]) for m in ["ARIMA", "Naiwna", "Mamdani", "LSTM", "TCN"]}}
        WYN["C_rezimy_ex_ante"][ds] = c_res
        print(f"   rezim wysoki (n={hi.sum()}): " + " ".join(f"{m}={c_res['MAPE_wysoka'][m]}" for m in ["TCN-Mamdani", "ARIMA", "Naiwna", "Mamdani"])
              + f" | DM hyb-ARIMA={c_res['DM_wysoka_hybryda_vs']['ARIMA']['DM']} NW={c_res['DM_wysoka_hybryda_vs']['ARIMA']['DM_NW']}")

    # ---------------- E, F, G, H (na pelnej probce hybrydy)
    data = prepare_data(ds)
    sc = data["scaler"]
    hdf = load_pred(*HYB[ds])
    dts = pd.to_datetime(data["dates_test"]).normalize()
    assert len(dts) == len(hdf) and (dts == hdf.index).all(), ds
    F = hdf[["TCN_trend", "TCN_volatility", "TCN_momentum"]].values
    sim = build_mamdani()
    pc = np.array([mamdani_predict(sim, f) for f in F])
    last_s = data["X_test"][:, -1]
    last_p = sc.inverse_transform(last_s.reshape(-1, 1)).ravel()
    rec = sc.inverse_transform((last_s * (1 + pc / 100)).reshape(-1, 1)).ravel()
    actual = hdf["Actual"].values
    pred_price = last_p * (1 + pc / 100)
    eff = (hdf["Predicted"].values / last_p - 1) * 100
    WYN["E_wariant_skali_cen"][ds] = {
        "zrodlo_prognoz": "/".join(HYB[ds]),
        "max_wzgl_blad_odtworzenia_prognoz": float(f"{np.max(np.abs(rec - hdf['Predicted'].values) / hdf['Predicted'].values):.2e}"),
        "MAPE_skala_znormalizowana_(opublikowany_wzor_kodu)": R(mape(actual, hdf["Predicted"].values)),
        "MAPE_skala_cen_(wzor_tekstowy)": R(mape(actual, pred_price)),
        "MAPE_naiwna_ta_sama_probka": R(mape(actual, last_p)),
        "sredni_wynik_Mamdaniego_pc_pct": R(pc.mean(), 4), "odch_std_pc": R(pc.std(), 4),
        "srednia_efektywna_zmiana_ceny_pct_dziennie": R(eff.mean(), 4),
        "odsetek_dni_prognoza_wzrostu_pct": R(100 * np.mean(hdf["Predicted"].values > last_p), 1),
    }
    tr = data["train_df"]["Close"].values
    WYN["F_poza_zakresem_treningu"][ds] = {
        "max_treningu": R(tr.max(), 4), "min_treningu": R(tr.min(), 4),
        "odsetek_dni_testu_powyzej_max_treningu_pct": R(100 * np.mean(actual > tr.max()), 1),
        "odsetek_dni_testu_ponizej_min_treningu_pct": R(100 * np.mean(actual < tr.min()), 1),
        "zakres_skalowanych_cen_testu": [R(sc.transform(actual.reshape(-1, 1)).min(), 3), R(sc.transform(actual.reshape(-1, 1)).max(), 3)],
    }

    def cr(x, y):
        r, p = stats.pearsonr(x, y)
        return {"r": R(r, 3), "p_nominalne": float(f"{p:.3g}")}
    WYN["G_korelacje"][ds] = {
        "trend_vs_poziom_ceny": cr(F[:, 0], last_p), "zmiennosc_vs_poziom_ceny": cr(F[:, 1], last_p),
        "momentum_vs_poziom_ceny": cr(F[:, 2], last_p), "trend_vs_momentum": cr(F[:, 0], F[:, 2]),
        "odch_std_wyjsc": {"trend": R(F[:, 0].std(), 3), "zmiennosc": R(F[:, 1].std(), 3), "momentum": R(F[:, 2].std(), 3)},
        "uwaga": "p nominalne: kolejne okna 30-dniowe nakladaja sie, obserwacje nie sa niezalezne",
    }
    S = rule_strengths(F)
    WYN["H_reguly"][ds] = {f"R{j+1}": {"srednia_sila": R(S[:, j].mean(), 3), "odsetek_dni_sila_>0.1_pct": R(100 * np.mean(S[:, j] > 0.1), 1)}
                           for j in range(15)}
    akt = [f"R{j+1}" for j in range(15) if np.mean(S[:, j] > 0.1) > 0.05]
    print(f"   E: MAPE kod={WYN['E_wariant_skali_cen'][ds]['MAPE_skala_znormalizowana_(opublikowany_wzor_kodu)']} "
          f"ceny={WYN['E_wariant_skali_cen'][ds]['MAPE_skala_cen_(wzor_tekstowy)']} naiwna={WYN['E_wariant_skali_cen'][ds]['MAPE_naiwna_ta_sama_probka']} "
          f"odtw={WYN['E_wariant_skali_cen'][ds]['max_wzgl_blad_odtworzenia_prognoz']} pc={WYN['E_wariant_skali_cen'][ds]['sredni_wynik_Mamdaniego_pc_pct']} "
          f"eff={WYN['E_wariant_skali_cen'][ds]['srednia_efektywna_zmiana_ceny_pct_dziennie']}")
    print(f"   F: >max={WYN['F_poza_zakresem_treningu'][ds]['odsetek_dni_testu_powyzej_max_treningu_pct']}% "
          f"<min={WYN['F_poza_zakresem_treningu'][ds]['odsetek_dni_testu_ponizej_min_treningu_pct']}% | "
          f"G: r(trend,poziom)={WYN['G_korelacje'][ds]['trend_vs_poziom_ceny']['r']} r(trend,mom)={WYN['G_korelacje'][ds]['trend_vs_momentum']['r']} "
          f"| H: aktywne (>5% dni z sila>0,1): {akt}")

# ---------------- I: prognoza naiwna na probkach eksperymentu 11
lb11 = json.load(open(os.path.join(BASE, "11_rozne_lookback", "wyniki.json"), encoding="utf-8"))
test_sp = prepare_data("SP500")["test_df"]["Close"].values
for lb in lb11["lookback_values"]:
    a_ = test_sp[lb:]
    WYN["I_lookback_naiwna_SP500"][f"lookback_{lb}"] = {
        "MAPE_hybryda_eksp11": lb11["wyniki"][f"lookback_{lb}"]["MAPE"],
        "MAPE_naiwna_ta_sama_probka": R(mape(a_, test_sp[lb - 1:-1])), "n": len(a_)}
print("\nI:", WYN["I_lookback_naiwna_SP500"])

# ---------------- J: prognoza naiwna na probce walidacyjnej S&P 500 (ta, na ktorej GA w eksp. 19 liczyl MAPE_val)
w19 = json.load(open(os.path.join(BASE, "19_ga_reguly_optymalizacja", "wyniki.json"), encoding="utf-8"))
val_sp = prepare_data("SP500")["val_df"]["Close"].values
WYN["J_walidacja_SP500_GA"] = {
    "MAPE_val_reczne_eksp19": w19["baseline_reczne"]["MAPE_val"],
    "MAPE_val_GA_eksp19": w19["ga_optymalizowane"]["MAPE_val"],
    "MAPE_val_naiwna": R(mape(val_sp[LOOKBACK:], val_sp[LOOKBACK - 1:-1])), "n": len(val_sp) - LOOKBACK}
print("J:", WYN["J_walidacja_SP500_GA"])

save_results({"eksperyment": "27: kontrole na wspolnych datach (koszty, rezimy ex ante, DM Newey-West, "
                             "wariant skali cen, zakres treningu, korelacje z poziomem, sila regul)",
              "zrodla_prognoz": {"ARIMA": "01", "LSTM": "02", "TCN": "03", "Mamdani": "04",
                                 "TCN-Mamdani": "05 (S&P 500, WIG20, EUR/USD, BTC/USD), 20 (DAX, Nikkei 225, Gold)",
                                 "ARIMA_stale i Naiwna": "23"},
              "wyniki": WYN}, os.path.join(OUT, "wyniki.json"))
print("\n=== EKSPERYMENT 27 ZAKONCZONY ===")
