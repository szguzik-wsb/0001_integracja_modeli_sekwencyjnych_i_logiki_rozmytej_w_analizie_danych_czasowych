# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 20: TCN-Mamdani hybrid na WSZYSTKICH 7 zbiorach danych.

Uruchamia architekture hybrydowa TCN + Mamdani (z eksperymentu 05) na pelnym
zestawie 7 zbiorow: S&P 500, WIG20, EUR/USD, BTC/USD, DAX, Nikkei 225, Gold.

Dodatkowo uruchamia ARIMA baseline na 3 nowych zbiorach (DAX, NIKKEI, GOLD)
w celu porownania.

Architektura TCN-Mamdani:
  [Dane cenowe] -> [TCN(channels=[32,32])]
                       |
                       v
                [3 cechy: trend, zmiennosc, momentum]
                       |
                       v
                [System Mamdaniego: 15 regul, Gaussian MF, COG]
                       |
                       v
                [Prognoza]
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from statsmodels.tsa.arima.model import ARIMA
from utils import (prepare_data, load_dataset, split_data,
                   compute_all_metrics, inverse_transform, save_results)
from config import DATASETS, LOOKBACK, EPOCHS, BATCH_SIZE, LEARNING_RATE, RANDOM_SEED

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# === TCN FEATURE EXTRACTOR (z eksperymentu 05) ===

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :-self.padding] if self.padding > 0 else out


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        if self.downsample:
            residual = self.downsample(residual)
        return self.relu(out + residual)


class TCNFeatureExtractor(nn.Module):
    """TCN ktory wyciaga 3 cechy: trend, zmiennosc, momentum."""
    def __init__(self, input_size=1, num_channels=[32, 32], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, 2 ** i, dropout))
        self.network = nn.Sequential(*layers)
        self.fc_trend = nn.Linear(num_channels[-1], 1)
        self.fc_volatility = nn.Linear(num_channels[-1], 1)
        self.fc_momentum = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = self.network(x)[:, :, -1]
        trend = torch.tanh(self.fc_trend(features)) * 3
        volatility = torch.sigmoid(self.fc_volatility(features)) * 5
        momentum = torch.tanh(self.fc_momentum(features)) * 3
        return torch.cat([trend, volatility, momentum], dim=1)


# === MAMDANI INFERENCE (z eksperymentu 05) ===

def build_mamdani():
    """System Mamdaniego z 3 wejsciami (cechy TCN) i 1 wyjsciem."""
    trend = ctrl.Antecedent(np.linspace(-3, 3, 1000), "trend")
    zmiennosc = ctrl.Antecedent(np.linspace(0, 5, 1000), "zmiennosc")
    momentum = ctrl.Antecedent(np.linspace(-3, 3, 1000), "momentum")
    prognoza = ctrl.Consequent(np.linspace(-5, 5, 1000), "prognoza")

    for var in [trend, momentum, prognoza]:
        var["duzy_spadek"] = fuzz.gaussmf(var.universe, -2.5, 0.6)
        var["spadek"] = fuzz.gaussmf(var.universe, -1, 0.4)
        var["neutralny"] = fuzz.gaussmf(var.universe, 0, 0.35)
        var["wzrost"] = fuzz.gaussmf(var.universe, 1, 0.4)
        var["duzy_wzrost"] = fuzz.gaussmf(var.universe, 2.5, 0.6)

    zmiennosc["niska"] = fuzz.gaussmf(zmiennosc.universe, 0, 0.5)
    zmiennosc["srednia"] = fuzz.gaussmf(zmiennosc.universe, 2, 0.6)
    zmiennosc["wysoka"] = fuzz.gaussmf(zmiennosc.universe, 4, 0.7)

    rules = [
        ctrl.Rule(trend["duzy_wzrost"] & momentum["duzy_wzrost"], prognoza["duzy_wzrost"]),
        ctrl.Rule(trend["wzrost"] & momentum["wzrost"], prognoza["wzrost"]),
        ctrl.Rule(trend["spadek"] & momentum["spadek"], prognoza["spadek"]),
        ctrl.Rule(trend["duzy_spadek"] & momentum["duzy_spadek"], prognoza["duzy_spadek"]),
        ctrl.Rule(trend["wzrost"] & momentum["spadek"], prognoza["neutralny"]),
        ctrl.Rule(trend["spadek"] & momentum["wzrost"], prognoza["neutralny"]),
        ctrl.Rule(trend["neutralny"] & momentum["neutralny"], prognoza["neutralny"]),
        ctrl.Rule(trend["neutralny"] & momentum["wzrost"], prognoza["wzrost"]),
        ctrl.Rule(trend["neutralny"] & momentum["spadek"], prognoza["spadek"]),
        ctrl.Rule(trend["duzy_wzrost"] & zmiennosc["wysoka"], prognoza["neutralny"]),
        ctrl.Rule(trend["duzy_spadek"] & zmiennosc["wysoka"], prognoza["neutralny"]),
        ctrl.Rule(trend["wzrost"] & zmiennosc["niska"], prognoza["duzy_wzrost"]),
        ctrl.Rule(trend["spadek"] & zmiennosc["niska"], prognoza["duzy_spadek"]),
        ctrl.Rule(momentum["duzy_wzrost"] & zmiennosc["srednia"], prognoza["wzrost"]),
        ctrl.Rule(momentum["duzy_spadek"] & zmiennosc["srednia"], prognoza["spadek"]),
    ]

    system = ctrl.ControlSystem(rules)
    return ctrl.ControlSystemSimulation(system)


def mamdani_predict(sim, tcn_features):
    """Prognoza Mamdaniego na podstawie cech TCN."""
    trend_val = np.clip(tcn_features[0], -2.9, 2.9)
    vol_val = np.clip(tcn_features[1], 0.01, 4.9)
    mom_val = np.clip(tcn_features[2], -2.9, 2.9)

    sim.input["trend"] = trend_val
    sim.input["zmiennosc"] = vol_val
    sim.input["momentum"] = mom_val

    try:
        sim.compute()
        return sim.output["prognoza"]
    except Exception:
        return 0.0


# === HYBRID TRAINING (z eksperymentu 05) ===

def train_hybrid(dataset_key):
    print(f"\n=== TCN-Mamdani: {DATASETS[dataset_key]['name']} ===")
    t0 = time.time()

    data = prepare_data(dataset_key)
    X_train = torch.FloatTensor(data["X_train"]).unsqueeze(-1)
    y_train = torch.FloatTensor(data["y_train"])
    X_val = torch.FloatTensor(data["X_val"]).unsqueeze(-1)
    y_val = torch.FloatTensor(data["y_val"])
    X_test = torch.FloatTensor(data["X_test"]).unsqueeze(-1)

    scaler = data["scaler"]
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)

    # Faza 1: Trenuj TCN feature extractor
    tcn = TCNFeatureExtractor()
    fc_proxy = nn.Linear(3, 1)

    params = list(tcn.parameters()) + list(fc_proxy.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience, patience_count = 15, 0

    print("  Faza 1: Trening TCN feature extractor...")
    for epoch in range(EPOCHS):
        tcn.train()
        fc_proxy.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            features = tcn(xb)
            pred = fc_proxy(features).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        tcn.eval()
        fc_proxy.eval()
        with torch.no_grad():
            val_feat = tcn(X_val)
            val_pred = fc_proxy(val_feat).squeeze()
            val_loss = criterion(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_tcn_state = tcn.state_dict().copy()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"    Early stopping epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}, val_loss: {val_loss:.6f}")

    tcn.load_state_dict(best_tcn_state)
    tcn.eval()

    # Faza 2: Ekstrakcja cech TCN + wnioskowanie Mamdaniego
    print("  Faza 2: Ekstrakcja cech TCN + wnioskowanie Mamdaniego...")
    mamdani_sim = build_mamdani()

    with torch.no_grad():
        test_features = tcn(X_test).numpy()

    # Faza 3: Wnioskowanie Mamdaniego
    predictions_scaled = []
    for i in range(len(test_features)):
        pred_change = mamdani_predict(mamdani_sim, test_features[i])
        last_val = data["X_test"][i, -1]
        pred_val = last_val * (1 + pred_change / 100)
        predictions_scaled.append(pred_val)

    predictions_scaled = np.array(predictions_scaled)
    predictions = inverse_transform(scaler, predictions_scaled)
    actuals = inverse_transform(scaler, data["y_test"])
    elapsed = time.time() - t0

    metrics = compute_all_metrics(actuals, predictions)
    metrics["czas_s"] = round(elapsed, 2)
    metrics["tcn_channels"] = [32, 32]
    metrics["mamdani_reguly"] = 15
    metrics["cechy_tcn"] = ["trend", "zmiennosc", "momentum"]

    print(f"  MAPE: {metrics['MAPE']}%")
    print(f"  RMSE: {metrics['RMSE']}")
    print(f"  MAE:  {metrics['MAE']}")
    print(f"  Czas: {metrics['czas_s']}s")

    # Zapisz prognozy
    pred_df = pd.DataFrame({
        "Actual": actuals,
        "Predicted": predictions,
        "TCN_trend": test_features[:, 0],
        "TCN_volatility": test_features[:, 1],
        "TCN_momentum": test_features[:, 2],
    })
    pred_df.to_csv(os.path.join(os.path.dirname(__file__),
                                f"prognozy_hybrid_{dataset_key}.csv"), index=False)

    return metrics


# === ARIMA BASELINE DLA NOWYCH ZBIOROW ===

NEW_DATASETS = ["DAX", "NIKKEI", "GOLD"]


def run_arima_baseline(dataset_key):
    """ARIMA(5,1,0) walk-forward na jednym zbiorze danych."""
    print(f"\n=== ARIMA: {DATASETS[dataset_key]['name']} ===")
    t0 = time.time()

    df = load_dataset(dataset_key)
    train_df, val_df, test_df = split_data(df)

    train_vals = train_df["Close"].values
    test_vals = test_df["Close"].values

    history = list(train_vals)
    history.extend(list(val_df["Close"].values))
    predictions = []

    for i in range(len(test_vals)):
        try:
            model = ARIMA(history, order=(5, 1, 0))
            fitted = model.fit()
            yhat = fitted.forecast(steps=1)[0]
        except Exception:
            yhat = history[-1]

        predictions.append(yhat)
        history.append(test_vals[i])

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(test_vals)}")

    predictions = np.array(predictions)
    elapsed = time.time() - t0

    metrics = compute_all_metrics(test_vals, predictions)
    metrics["czas_s"] = round(elapsed, 2)

    print(f"  MAPE: {metrics['MAPE']}%")
    print(f"  RMSE: {metrics['RMSE']}")
    print(f"  MAE:  {metrics['MAE']}")
    print(f"  Czas: {metrics['czas_s']}s")

    pred_df = pd.DataFrame({
        "Actual": test_vals,
        "Predicted": predictions,
    })
    pred_df.to_csv(os.path.join(os.path.dirname(__file__),
                                f"prognozy_arima_{dataset_key}.csv"), index=False)

    return metrics


# === GLOWNA PETLA ===

print("=" * 60)
print("  EKSPERYMENT 20: TCN-MAMDANI HYBRID NA 7 ZBIORACH DANYCH")
print("=" * 60)

# 1. TCN-Mamdani na wszystkich 7 zbiorach
HYBRID_RESULTS = {}
for ds_key in DATASETS:
    # Reset seed dla kazdego zbioru (reprodukowalnosc)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    HYBRID_RESULTS[ds_key] = train_hybrid(ds_key)

# 2. ARIMA na 3 nowych zbiorach
print("\n" + "=" * 60)
print("  ARIMA BASELINE NA NOWYCH ZBIORACH")
print("=" * 60)

ARIMA_RESULTS = {}
for ds_key in NEW_DATASETS:
    ARIMA_RESULTS[ds_key] = run_arima_baseline(ds_key)

# 3. Podsumowanie porownawcze
print("\n" + "=" * 60)
print("  PODSUMOWANIE")
print("=" * 60)

print(f"\n  {'Zbior':<12} {'TCN-Mamdani MAPE':>18} {'ARIMA MAPE':>12}")
print("  " + "-" * 44)
for ds_key in DATASETS:
    hybrid_mape = HYBRID_RESULTS[ds_key]["MAPE"]
    if ds_key in ARIMA_RESULTS:
        arima_mape = ARIMA_RESULTS[ds_key]["MAPE"]
        print(f"  {ds_key:<12} {hybrid_mape:>17.4f}% {arima_mape:>11.4f}%")
    else:
        print(f"  {ds_key:<12} {hybrid_mape:>17.4f}%       {'—':>5}")

# 4. Zapis wynikow
result_data = {
    "model": "TCN(channels=[32,32]) + Mamdani(15 regul, gaussMF, COG)",
    "architektura": "TCN feature extractor -> fuzzyfikacja -> Mamdani inference -> defuzyfikacja COG",
    "zbiory_danych": list(DATASETS.keys()),
    "wyniki_hybrid": HYBRID_RESULTS,
    "wyniki_arima_nowe": ARIMA_RESULTS,
    "arima_order": "(5,1,0)",
    "reguly_mamdaniego": [
        "R1:  IF trend=duzy_wzrost AND momentum=duzy_wzrost THEN prognoza=duzy_wzrost",
        "R2:  IF trend=wzrost AND momentum=wzrost THEN prognoza=wzrost",
        "R3:  IF trend=spadek AND momentum=spadek THEN prognoza=spadek",
        "R4:  IF trend=duzy_spadek AND momentum=duzy_spadek THEN prognoza=duzy_spadek",
        "R5:  IF trend=wzrost AND momentum=spadek THEN prognoza=neutralny",
        "R6:  IF trend=spadek AND momentum=wzrost THEN prognoza=neutralny",
        "R7:  IF trend=neutralny AND momentum=neutralny THEN prognoza=neutralny",
        "R8:  IF trend=neutralny AND momentum=wzrost THEN prognoza=wzrost",
        "R9:  IF trend=neutralny AND momentum=spadek THEN prognoza=spadek",
        "R10: IF trend=duzy_wzrost AND zmiennosc=wysoka THEN prognoza=neutralny",
        "R11: IF trend=duzy_spadek AND zmiennosc=wysoka THEN prognoza=neutralny",
        "R12: IF trend=wzrost AND zmiennosc=niska THEN prognoza=duzy_wzrost",
        "R13: IF trend=spadek AND zmiennosc=niska THEN prognoza=duzy_spadek",
        "R14: IF momentum=duzy_wzrost AND zmiennosc=srednia THEN prognoza=wzrost",
        "R15: IF momentum=duzy_spadek AND zmiennosc=srednia THEN prognoza=spadek",
    ],
}
save_results(result_data, os.path.join(os.path.dirname(__file__), "wyniki.json"))
print("\n=== EKSPERYMENT 20 ZAKONCZONE ===")
