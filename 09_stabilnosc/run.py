# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 09: Analiza stabilnosci wynikow TCN+Mamdani.

Cel: Wykazanie, ze wyniki modelu hybrydowego nie sa przypadkowe.
Model TCN+Mamdani jest uruchamiany 5 razy z roznymi ziarnami losowosci
(seeds: 42, 123, 456, 789, 1024) na zbiorze S&P 500.

Dla kazdego uruchomienia rejestrowane sa MAPE, RMSE, MAE.
Obliczane sa srednia i odchylenie standardowe kazdej metryki.
Niskie odchylenie standardowe swiadczy o stabilnosci modelu.
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
from config import DATASETS, LOOKBACK, EPOCHS, BATCH_SIZE, LEARNING_RATE

SEEDS = [42, 123, 456, 789, 1024]
DATASET_KEY = "SP500"
EXP_DIR = os.path.dirname(os.path.abspath(__file__))


# === TCN + MAMDANI (z 05_tcn_mamdani_hybrid) ===

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
    def __init__(self, input_size=1, num_channels=[32, 32],
                 kernel_size=3, dropout=0.2):
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


def build_mamdani():
    """System Mamdaniego z 3 wejsciami i 1 wyjsciem."""
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
    sim.input["trend"] = np.clip(tcn_features[0], -2.9, 2.9)
    sim.input["zmiennosc"] = np.clip(tcn_features[1], 0.01, 4.9)
    sim.input["momentum"] = np.clip(tcn_features[2], -2.9, 2.9)
    try:
        sim.compute()
        return sim.output["prognoza"]
    except Exception:
        return 0.0


# === POJEDYNCZE URUCHOMIENIE ===

def run_single(seed):
    """Uruchom TCN+Mamdani z danym ziarnem losowosci."""
    print(f"\n  Seed={seed}")
    t0 = time.time()

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data = prepare_data(DATASET_KEY)
    X_train = torch.FloatTensor(data["X_train"]).unsqueeze(-1)
    y_train = torch.FloatTensor(data["y_train"])
    X_val = torch.FloatTensor(data["X_val"]).unsqueeze(-1)
    y_val = torch.FloatTensor(data["y_val"])
    X_test = torch.FloatTensor(data["X_test"]).unsqueeze(-1)

    scaler = data["scaler"]
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))

    # Trening TCN
    tcn = TCNFeatureExtractor()
    fc_proxy = nn.Linear(3, 1)
    params = list(tcn.parameters()) + list(fc_proxy.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience, patience_count = 15, 0

    for epoch in range(EPOCHS):
        tcn.train(); fc_proxy.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            features = tcn(xb)
            pred = fc_proxy(features).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        tcn.eval(); fc_proxy.eval()
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
                break

    tcn.load_state_dict(best_tcn_state)
    tcn.eval()

    # Wnioskowanie Mamdaniego
    mamdani_sim = build_mamdani()

    with torch.no_grad():
        test_features = tcn(X_test).numpy()

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
    metrics["seed"] = seed

    print(f"    MAPE: {metrics['MAPE']}%")
    print(f"    RMSE: {metrics['RMSE']}")
    print(f"    MAE:  {metrics['MAE']}")
    print(f"    Czas: {metrics['czas_s']}s")

    return metrics


# === GLOWNA PETLA ===

print("=" * 60)
print("EKSPERYMENT 09: ANALIZA STABILNOSCI WYNIKOW")
print(f"Zbior danych: {DATASETS[DATASET_KEY]['name']}")
print(f"Seeds: {SEEDS}")
print(f"Liczba uruchomien: {len(SEEDS)}")
print("=" * 60)

per_run = []
for seed in SEEDS:
    metrics = run_single(seed)
    per_run.append(metrics)

# Statystyki zbiorcze
mapes = [r["MAPE"] for r in per_run]
rmses = [r["RMSE"] for r in per_run]
maes = [r["MAE"] for r in per_run]

aggregate = {
    "MAPE": {
        "mean": round(float(np.mean(mapes)), 4),
        "std": round(float(np.std(mapes)), 4),
        "min": round(float(np.min(mapes)), 4),
        "max": round(float(np.max(mapes)), 4),
    },
    "RMSE": {
        "mean": round(float(np.mean(rmses)), 6),
        "std": round(float(np.std(rmses)), 6),
        "min": round(float(np.min(rmses)), 6),
        "max": round(float(np.max(rmses)), 6),
    },
    "MAE": {
        "mean": round(float(np.mean(maes)), 6),
        "std": round(float(np.std(maes)), 6),
        "min": round(float(np.min(maes)), 6),
        "max": round(float(np.max(maes)), 6),
    },
}

# Podsumowanie
print("\n" + "=" * 60)
print("PODSUMOWANIE STABILNOSCI:")
print(f"  MAPE: {aggregate['MAPE']['mean']} +/- {aggregate['MAPE']['std']}%")
print(f"  RMSE: {aggregate['RMSE']['mean']} +/- {aggregate['RMSE']['std']}")
print(f"  MAE:  {aggregate['MAE']['mean']} +/- {aggregate['MAE']['std']}")
print(f"  Zakres MAPE: [{aggregate['MAPE']['min']}, {aggregate['MAPE']['max']}]%")

results = {
    "opis": "Analiza stabilnosci TCN+Mamdani: 5 uruchomien z roznymi seeds",
    "zbior_danych": DATASET_KEY,
    "seeds": SEEDS,
    "wyniki_per_run": per_run,
    "statystyki_zbiorcze": aggregate,
}

save_results(results, os.path.join(EXP_DIR, "wyniki.json"))
print("\n=== EKSPERYMENT 09 ZAKONCZONY ===")
