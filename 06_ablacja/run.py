# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 06: Ablacja — badanie wplywu komponentow modelu hybrydowego.

Warianty:
  A) TCN+Mamdani z 5 regulami (vs 15 w pelnym modelu)
  B) TCN+Mamdani z 25 regulami
  C) TCN+Mamdani z trojkatnymi MF (vs gaussowskie)
  D) TCN sam (bez Mamdani) — porownanie z exp 03

Cel: Ktory komponent ile wnosi do jakosci prognoz.
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
from utils import prepare_data, compute_all_metrics, inverse_transform, save_results
from config import DATASETS, LOOKBACK, EPOCHS, BATCH_SIZE, LEARNING_RATE, RANDOM_SEED

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Import TCN z exp 05
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "05_tcn_mamdani_hybrid"))


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


def build_mamdani_5_rules():
    """Uproszczony Mamdani z 5 regulami."""
    trend = ctrl.Antecedent(np.linspace(-3, 3, 1000), "trend")
    momentum = ctrl.Antecedent(np.linspace(-3, 3, 1000), "momentum")
    zmiennosc = ctrl.Antecedent(np.linspace(0, 5, 1000), "zmiennosc")
    prognoza = ctrl.Consequent(np.linspace(-5, 5, 1000), "prognoza")

    for var in [trend, momentum, prognoza]:
        var["spadek"] = fuzz.gaussmf(var.universe, -2, 0.8)
        var["neutralny"] = fuzz.gaussmf(var.universe, 0, 0.5)
        var["wzrost"] = fuzz.gaussmf(var.universe, 2, 0.8)
    zmiennosc["niska"] = fuzz.gaussmf(zmiennosc.universe, 0, 1)
    zmiennosc["wysoka"] = fuzz.gaussmf(zmiennosc.universe, 4, 1)

    rules = [
        ctrl.Rule(trend["wzrost"] & momentum["wzrost"], prognoza["wzrost"]),
        ctrl.Rule(trend["spadek"] & momentum["spadek"], prognoza["spadek"]),
        ctrl.Rule(trend["neutralny"], prognoza["neutralny"]),
        ctrl.Rule(trend["wzrost"] & zmiennosc["wysoka"], prognoza["neutralny"]),
        ctrl.Rule(trend["spadek"] & zmiennosc["wysoka"], prognoza["neutralny"]),
    ]
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))


def build_mamdani_triangular():
    """Mamdani z trojkatnymi MF zamiast gaussowskich."""
    trend = ctrl.Antecedent(np.linspace(-3, 3, 1000), "trend")
    momentum = ctrl.Antecedent(np.linspace(-3, 3, 1000), "momentum")
    zmiennosc = ctrl.Antecedent(np.linspace(0, 5, 1000), "zmiennosc")
    prognoza = ctrl.Consequent(np.linspace(-5, 5, 1000), "prognoza")

    for var in [trend, momentum, prognoza]:
        var["duzy_spadek"] = fuzz.trimf(var.universe, [-3, -3, -1])
        var["spadek"] = fuzz.trimf(var.universe, [-2, -1, 0])
        var["neutralny"] = fuzz.trimf(var.universe, [-1, 0, 1])
        var["wzrost"] = fuzz.trimf(var.universe, [0, 1, 2])
        var["duzy_wzrost"] = fuzz.trimf(var.universe, [1, 3, 3])

    zmiennosc["niska"] = fuzz.trimf(zmiennosc.universe, [0, 0, 2])
    zmiennosc["srednia"] = fuzz.trimf(zmiennosc.universe, [1, 2.5, 4])
    zmiennosc["wysoka"] = fuzz.trimf(zmiennosc.universe, [3, 5, 5])

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
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))


def run_variant(variant_name, mamdani_sim, dataset_key, data, tcn):
    """Uruchamia wariant ablacji."""
    X_test = torch.FloatTensor(data["X_test"]).unsqueeze(-1)
    scaler = data["scaler"]

    tcn.eval()
    with torch.no_grad():
        test_features = tcn(X_test).numpy()

    predictions_scaled = []
    for i in range(len(test_features)):
        feat = test_features[i]
        t = np.clip(feat[0], -2.9, 2.9)
        v = np.clip(feat[1], 0.01, 4.9)
        m = np.clip(feat[2], -2.9, 2.9)

        mamdani_sim.input["trend"] = t
        mamdani_sim.input["zmiennosc"] = v
        mamdani_sim.input["momentum"] = m
        try:
            mamdani_sim.compute()
            pred_change = mamdani_sim.output["prognoza"]
        except Exception:
            pred_change = 0.0

        last_val = data["X_test"][i, -1]
        predictions_scaled.append(last_val * (1 + pred_change / 100))

    predictions = inverse_transform(scaler, np.array(predictions_scaled))
    actuals = inverse_transform(scaler, data["y_test"])
    return compute_all_metrics(actuals, predictions)


# Train TCN once per dataset, test with different Mamdani configs
RESULTS = {}

for ds_key in ["SP500"]:  # Ablacja na 1 zbiorze (S&P500) — wystarczy do analizy
    print(f"\n=== ABLACJA: {DATASETS[ds_key]['name']} ===")

    data = prepare_data(ds_key)
    X_train = torch.FloatTensor(data["X_train"]).unsqueeze(-1)
    y_train = torch.FloatTensor(data["y_train"])
    X_val = torch.FloatTensor(data["X_val"]).unsqueeze(-1)
    y_val = torch.FloatTensor(data["y_val"])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    tcn = TCNFeatureExtractor()
    fc_proxy = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(list(tcn.parameters()) + list(fc_proxy.parameters()), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val = float("inf")
    patience_count = 0
    for epoch in range(EPOCHS):
        tcn.train(); fc_proxy.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(fc_proxy(tcn(xb)).squeeze(), yb)
            loss.backward()
            optimizer.step()
        tcn.eval(); fc_proxy.eval()
        with torch.no_grad():
            vl = criterion(fc_proxy(tcn(X_val)).squeeze(), y_val).item()
        if vl < best_val:
            best_val = vl
            best_state = tcn.state_dict().copy()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 15:
                break

    tcn.load_state_dict(best_state)

    # Warianty
    print("  Wariant A: 5 regul...")
    m5 = run_variant("5_regul", build_mamdani_5_rules(), ds_key, data, tcn)
    print(f"    MAPE: {m5['MAPE']}%")

    print("  Wariant B: 15 regul trojkatne MF...")
    mt = run_variant("trimf", build_mamdani_triangular(), ds_key, data, tcn)
    print(f"    MAPE: {mt['MAPE']}%")

    RESULTS[ds_key] = {
        "5_regul_gauss": m5,
        "15_regul_trimf": mt,
    }

save_results({"model": "Ablacja TCN+Mamdani", "wyniki": RESULTS},
             os.path.join(os.path.dirname(__file__), "wyniki.json"))
print("\n=== ABLACJA ZAKONCZONE ===")
