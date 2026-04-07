# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""Eksperyment 04: System Mamdaniego baseline na 4 zbiorach danych."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from utils import load_dataset, split_data, compute_all_metrics, save_results
from config import DATASETS, LOOKBACK


def build_mamdani_system():
    """Buduje system Mamdaniego dla prognozowania cen.

    Wejscia:
      - zmiana: procentowa zmiana ceny (lookback)
      - trend: sredni kierunek (narastajacy/malejacy)
      - zmiennosc: odchylenie standardowe zmian
    Wyjscie:
      - prognoza_zmiana: prognozowana procentowa zmiana
    """
    # Zmienne wejsciowe
    zmiana = ctrl.Antecedent(np.linspace(-5, 5, 1000), "zmiana")
    trend = ctrl.Antecedent(np.linspace(-3, 3, 1000), "trend")
    zmiennosc = ctrl.Antecedent(np.linspace(0, 5, 1000), "zmiennosc")

    # Zmienna wyjsciowa
    prognoza = ctrl.Consequent(np.linspace(-5, 5, 1000), "prognoza")

    # Funkcje przynaleznosci — gaussowskie
    for var in [zmiana, trend, prognoza]:
        var["duzy_spadek"] = fuzz.gaussmf(var.universe, -3, 0.8)
        var["spadek"] = fuzz.gaussmf(var.universe, -1.2, 0.5)
        var["neutralny"] = fuzz.gaussmf(var.universe, 0, 0.4)
        var["wzrost"] = fuzz.gaussmf(var.universe, 1.2, 0.5)
        var["duzy_wzrost"] = fuzz.gaussmf(var.universe, 3, 0.8)

    zmiennosc["niska"] = fuzz.gaussmf(zmiennosc.universe, 0, 0.5)
    zmiennosc["srednia"] = fuzz.gaussmf(zmiennosc.universe, 1.5, 0.5)
    zmiennosc["wysoka"] = fuzz.gaussmf(zmiennosc.universe, 3.5, 0.8)

    # Baza regul (15 regul)
    rules = [
        ctrl.Rule(zmiana["duzy_wzrost"] & trend["duzy_wzrost"], prognoza["wzrost"]),
        ctrl.Rule(zmiana["wzrost"] & trend["wzrost"], prognoza["wzrost"]),
        ctrl.Rule(zmiana["wzrost"] & trend["neutralny"], prognoza["neutralny"]),
        ctrl.Rule(zmiana["neutralny"] & trend["wzrost"], prognoza["neutralny"]),
        ctrl.Rule(zmiana["neutralny"] & trend["neutralny"], prognoza["neutralny"]),
        ctrl.Rule(zmiana["neutralny"] & trend["spadek"], prognoza["neutralny"]),
        ctrl.Rule(zmiana["spadek"] & trend["neutralny"], prognoza["neutralny"]),
        ctrl.Rule(zmiana["spadek"] & trend["spadek"], prognoza["spadek"]),
        ctrl.Rule(zmiana["duzy_spadek"] & trend["duzy_spadek"], prognoza["spadek"]),
        ctrl.Rule(zmiana["duzy_wzrost"] & zmiennosc["wysoka"], prognoza["neutralny"]),
        ctrl.Rule(zmiana["duzy_spadek"] & zmiennosc["wysoka"], prognoza["neutralny"]),
        ctrl.Rule(zmiana["wzrost"] & zmiennosc["niska"], prognoza["wzrost"]),
        ctrl.Rule(zmiana["spadek"] & zmiennosc["niska"], prognoza["spadek"]),
        ctrl.Rule(zmiana["duzy_wzrost"] & zmiennosc["niska"], prognoza["duzy_wzrost"]),
        ctrl.Rule(zmiana["duzy_spadek"] & zmiennosc["niska"], prognoza["duzy_spadek"]),
    ]

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim


def extract_features(prices, lookback=LOOKBACK):
    """Ekstrakcja cech z okna cenowego dla systemu Mamdaniego."""
    returns = np.diff(prices) / prices[:-1] * 100
    zmiana = returns[-1] if len(returns) > 0 else 0
    trend = np.mean(returns[-5:]) if len(returns) >= 5 else np.mean(returns)
    zmiennosc = np.std(returns) if len(returns) > 1 else 0

    # Ogranicz do zakresu zmiennych
    zmiana = np.clip(zmiana, -4.9, 4.9)
    trend = np.clip(trend, -2.9, 2.9)
    zmiennosc = np.clip(zmiennosc, 0.01, 4.9)

    return zmiana, trend, zmiennosc


def predict_mamdani(sim, prices, lookback=LOOKBACK):
    """Prognoza jednego kroku na podstawie okna cenowego."""
    zmiana, trend, zmiennosc = extract_features(prices, lookback)

    sim.input["zmiana"] = zmiana
    sim.input["trend"] = trend
    sim.input["zmiennosc"] = zmiennosc

    try:
        sim.compute()
        pred_change = sim.output["prognoza"]
    except Exception:
        pred_change = 0.0

    predicted_price = prices[-1] * (1 + pred_change / 100)
    return predicted_price


RESULTS = {}
sim = build_mamdani_system()

for ds_key, ds_info in DATASETS.items():
    print(f"\n=== {ds_info['name']} ===")
    t0 = time.time()

    df = load_dataset(ds_key)
    train_df, val_df, test_df = split_data(df)

    all_prices = pd.concat([train_df, val_df, test_df])["Close"].values
    test_start_idx = len(train_df) + len(val_df)

    predictions = []
    actuals = []

    for i in range(len(test_df)):
        idx = test_start_idx + i
        window = all_prices[max(0, idx - LOOKBACK):idx + 1]

        if len(window) < 3:
            pred = window[-1]
        else:
            pred = predict_mamdani(sim, window)

        if i < len(test_df) - 1:
            actual_next = all_prices[idx + 1]
            predictions.append(pred)
            actuals.append(actual_next)

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(test_df)}")

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    elapsed = time.time() - t0

    metrics = compute_all_metrics(actuals, predictions)
    metrics["czas_s"] = round(elapsed, 2)
    metrics["liczba_regul"] = 15
    RESULTS[ds_key] = metrics

    print(f"  MAPE: {metrics['MAPE']}%")
    print(f"  RMSE: {metrics['RMSE']}")
    print(f"  MAE:  {metrics['MAE']}")
    print(f"  Czas: {metrics['czas_s']}s")

    pred_df = pd.DataFrame({"Actual": actuals, "Predicted": predictions})
    pred_df.to_csv(os.path.join(os.path.dirname(__file__), f"prognozy_{ds_key}.csv"), index=False)

save_results({"model": "Mamdani(15 regul, gaussMF, COG)", "wyniki": RESULTS},
             os.path.join(os.path.dirname(__file__), "wyniki.json"))
print("\n=== MAMDANI BASELINE ZAKONCZONE ===")
