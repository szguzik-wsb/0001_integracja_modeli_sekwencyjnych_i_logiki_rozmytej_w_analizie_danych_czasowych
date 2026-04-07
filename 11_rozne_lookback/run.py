"""
Eksperyment 11: Wplyw rozmiaru okna wejsciowego (lookback) na jakosc prognoz TCN+Mamdani.

Testowane wartosci lookback: [10, 20, 30, 60, 90]
Zbior danych: S&P 500
Metryki: MAPE, RMSE, MAE
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
from config import DATASETS, EPOCHS, BATCH_SIZE, LEARNING_RATE, RANDOM_SEED

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# === TCN FEATURE EXTRACTOR ===

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


# === MAMDANI INFERENCE ===

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


# === EKSPERYMENT: ROZNE LOOKBACK ===

LOOKBACK_VALUES = [10, 20, 30, 60, 90]
DATASET_KEY = "SP500"

RESULTS = {}

for lookback in LOOKBACK_VALUES:
    print(f"\n=== Lookback = {lookback} ===")

    # Reset seedow dla kazdego lookback
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    t0 = time.time()

    data = prepare_data(DATASET_KEY, lookback=lookback)
    X_train = torch.FloatTensor(data["X_train"]).unsqueeze(-1)
    y_train = torch.FloatTensor(data["y_train"])
    X_val = torch.FloatTensor(data["X_val"]).unsqueeze(-1)
    y_val = torch.FloatTensor(data["y_val"])
    X_test = torch.FloatTensor(data["X_test"]).unsqueeze(-1)

    scaler = data["scaler"]
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)

    # Faza 1: Trening TCN
    tcn = TCNFeatureExtractor()
    fc_proxy = nn.Linear(3, 1)

    params = list(tcn.parameters()) + list(fc_proxy.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience, patience_count = 15, 0
    best_tcn_state = None

    print(f"  Faza 1: Trening TCN (lookback={lookback})...")
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

    # Faza 2: Ekstrakcja cech + Mamdani
    print(f"  Faza 2: Ekstrakcja cech TCN + wnioskowanie Mamdaniego...")
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
    metrics["lookback"] = lookback
    metrics["n_train"] = len(data["X_train"])
    metrics["n_test"] = len(data["X_test"])

    RESULTS[f"lookback_{lookback}"] = metrics

    print(f"  MAPE: {metrics['MAPE']}%")
    print(f"  RMSE: {metrics['RMSE']}")
    print(f"  MAE:  {metrics['MAE']}")
    print(f"  Czas: {metrics['czas_s']}s")

# Podsumowanie
print("\n=== PODSUMOWANIE ===")
print(f"{'Lookback':>10} {'MAPE':>10} {'RMSE':>12} {'MAE':>12} {'Czas(s)':>10}")
print("-" * 60)
for lb in LOOKBACK_VALUES:
    m = RESULTS[f"lookback_{lb}"]
    print(f"{lb:>10} {m['MAPE']:>10.4f} {m['RMSE']:>12.6f} {m['MAE']:>12.6f} {m['czas_s']:>10.2f}")

# Najlepszy lookback
best_lb = min(LOOKBACK_VALUES, key=lambda lb: RESULTS[f"lookback_{lb}"]["MAPE"])
print(f"\nNajlepszy lookback wg MAPE: {best_lb}")

result_data = {
    "eksperyment": "Wplyw rozmiaru okna wejsciowego (lookback) na TCN+Mamdani",
    "dataset": "SP500",
    "lookback_values": LOOKBACK_VALUES,
    "najlepszy_lookback_MAPE": best_lb,
    "wyniki": RESULTS,
}
save_results(result_data, os.path.join(os.path.dirname(__file__), "wyniki.json"))
print("\n=== EKSPERYMENT 11 ZAKONCZONE ===")
