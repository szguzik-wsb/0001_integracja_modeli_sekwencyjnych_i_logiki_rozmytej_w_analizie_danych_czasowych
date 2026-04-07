# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 13: Porownanie zlozonosci obliczeniowej 5 modeli.

Mierzone:
  - Czas treningu (s)
  - Czas inferencji na zbiorze testowym (s)
  - Zuzycie pamieci (MB)
  - Liczba parametrow (dla modeli DL)
Modele: ARIMA, LSTM, TCN, Mamdani, TCN+Mamdani
Zbior: S&P 500
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
from utils import (load_dataset, split_data, prepare_data, compute_all_metrics,
                   inverse_transform, save_results)
from config import DATASETS, LOOKBACK, EPOCHS, BATCH_SIZE, LEARNING_RATE, RANDOM_SEED

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATASET_KEY = "SP500"


# === MODELE ===

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()


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


class TCNModel(nn.Module):
    def __init__(self, input_size=1, num_channels=[32, 32, 32], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, 2 ** i, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)[:, :, -1]
        return self.fc(out).squeeze()


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


def build_mamdani():
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


def build_mamdani_baseline():
    """System Mamdaniego baseline (z eksperymentu 04)."""
    zmiana = ctrl.Antecedent(np.linspace(-5, 5, 1000), "zmiana")
    trend = ctrl.Antecedent(np.linspace(-3, 3, 1000), "trend")
    zmiennosc = ctrl.Antecedent(np.linspace(0, 5, 1000), "zmiennosc")
    prognoza = ctrl.Consequent(np.linspace(-5, 5, 1000), "prognoza")

    for var in [zmiana, trend, prognoza]:
        var["duzy_spadek"] = fuzz.gaussmf(var.universe, -3, 0.8)
        var["spadek"] = fuzz.gaussmf(var.universe, -1.2, 0.5)
        var["neutralny"] = fuzz.gaussmf(var.universe, 0, 0.4)
        var["wzrost"] = fuzz.gaussmf(var.universe, 1.2, 0.5)
        var["duzy_wzrost"] = fuzz.gaussmf(var.universe, 3, 0.8)

    zmiennosc["niska"] = fuzz.gaussmf(zmiennosc.universe, 0, 0.5)
    zmiennosc["srednia"] = fuzz.gaussmf(zmiennosc.universe, 1.5, 0.5)
    zmiennosc["wysoka"] = fuzz.gaussmf(zmiennosc.universe, 3.5, 0.8)

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
    return ctrl.ControlSystemSimulation(system)


def extract_features_baseline(prices, lookback=LOOKBACK):
    """Ekstrakcja cech dla Mamdani baseline."""
    returns = np.diff(prices) / prices[:-1] * 100
    zmiana = returns[-1] if len(returns) > 0 else 0
    trend = np.mean(returns[-5:]) if len(returns) >= 5 else np.mean(returns)
    zmiennosc = np.std(returns) if len(returns) > 1 else 0
    return np.clip(zmiana, -4.9, 4.9), np.clip(trend, -2.9, 2.9), np.clip(zmiennosc, 0.01, 4.9)


def count_parameters(model):
    """Zlicza parametry modelu PyTorch."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_memory_mb(model):
    """Szacuje zuzycie pamieci modelu w MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return round((param_size + buffer_size) / (1024 ** 2), 4)


def train_dl_model(model, train_loader, X_val, y_val, model_name):
    """Trenuje model DL i zwraca czas treningu."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float("inf")
    patience, patience_count = 15, 0
    best_state = None

    t_train_start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    t_train = time.time() - t_train_start
    model.load_state_dict(best_state)
    model.eval()
    return t_train


# === PRZYGOTOWANIE DANYCH ===

print(f"=== Eksperyment 13: Zlozonosc obliczeniowa ({DATASETS[DATASET_KEY]['name']}) ===\n")

data = prepare_data(DATASET_KEY)
X_train = torch.FloatTensor(data["X_train"]).unsqueeze(-1)
y_train = torch.FloatTensor(data["y_train"])
X_val = torch.FloatTensor(data["X_val"]).unsqueeze(-1)
y_val = torch.FloatTensor(data["y_val"])
X_test = torch.FloatTensor(data["X_test"]).unsqueeze(-1)
scaler = data["scaler"]

train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)

df = load_dataset(DATASET_KEY)
train_df, val_df, test_df = split_data(df)
all_prices = pd.concat([train_df, val_df, test_df])["Close"].values
test_start_idx = len(train_df) + len(val_df)

RESULTS = {}

# === 1. ARIMA ===
print("--- ARIMA(5,1,0) ---")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

history = list(train_df["Close"].values) + list(val_df["Close"].values)
test_vals = test_df["Close"].values

# Mierzenie czas treningu (= dopasowanie na calej historii)
t_train_start = time.time()
try:
    arima_model = ARIMA(history, order=(5, 1, 0))
    arima_fitted = arima_model.fit()
except Exception:
    arima_fitted = None
t_arima_train = time.time() - t_train_start

# Mierzenie czasu inferencji (walk-forward, 50 krokow)
n_inference = min(50, len(test_vals))
t_inf_start = time.time()
for i in range(n_inference):
    try:
        m = ARIMA(history[:len(history)], order=(5, 1, 0))
        f = m.fit()
        _ = f.forecast(steps=1)[0]
    except Exception:
        pass
t_arima_inf = time.time() - t_inf_start
t_arima_inf_per_sample = t_arima_inf / n_inference

# ARIMA nie ma "parametrow" w sensie DL — raportujemy rzad modelu
arima_memory = sys.getsizeof(arima_fitted) / (1024 ** 2) if arima_fitted else 0

RESULTS["ARIMA"] = {
    "czas_treningu_s": round(t_arima_train, 4),
    "czas_inferencji_s": round(t_arima_inf, 4),
    "czas_inferencji_na_probke_s": round(t_arima_inf_per_sample, 6),
    "n_probek_inferencji": n_inference,
    "parametry_total": "N/A (model statystyczny, rzad (5,1,0))",
    "parametry_trenowalnych": "N/A",
    "pamiec_modelu_MB": round(arima_memory, 4),
}
print(f"  Trening: {t_arima_train:.4f}s | Inferencja ({n_inference} probek): {t_arima_inf:.4f}s")

# === 2. LSTM ===
print("\n--- LSTM(hidden=64, layers=2) ---")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

lstm = LSTMModel()
total_p, train_p = count_parameters(lstm)
lstm_memory = get_model_memory_mb(lstm)

t_lstm_train = train_dl_model(lstm, train_loader, X_val, y_val, "LSTM")

t_inf_start = time.time()
with torch.no_grad():
    _ = lstm(X_test).numpy()
t_lstm_inf = time.time() - t_inf_start

RESULTS["LSTM"] = {
    "czas_treningu_s": round(t_lstm_train, 4),
    "czas_inferencji_s": round(t_lstm_inf, 4),
    "czas_inferencji_na_probke_s": round(t_lstm_inf / len(X_test), 6),
    "n_probek_inferencji": len(X_test),
    "parametry_total": total_p,
    "parametry_trenowalnych": train_p,
    "pamiec_modelu_MB": lstm_memory,
}
print(f"  Trening: {t_lstm_train:.4f}s | Inferencja: {t_lstm_inf:.4f}s | Parametry: {total_p}")

# === 3. TCN ===
print("\n--- TCN(channels=[32,32,32]) ---")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

tcn = TCNModel()
total_p, train_p = count_parameters(tcn)
tcn_memory = get_model_memory_mb(tcn)

t_tcn_train = train_dl_model(tcn, train_loader, X_val, y_val, "TCN")

t_inf_start = time.time()
with torch.no_grad():
    _ = tcn(X_test).numpy()
t_tcn_inf = time.time() - t_inf_start

RESULTS["TCN"] = {
    "czas_treningu_s": round(t_tcn_train, 4),
    "czas_inferencji_s": round(t_tcn_inf, 4),
    "czas_inferencji_na_probke_s": round(t_tcn_inf / len(X_test), 6),
    "n_probek_inferencji": len(X_test),
    "parametry_total": total_p,
    "parametry_trenowalnych": train_p,
    "pamiec_modelu_MB": tcn_memory,
}
print(f"  Trening: {t_tcn_train:.4f}s | Inferencja: {t_tcn_inf:.4f}s | Parametry: {total_p}")

# === 4. Mamdani baseline ===
print("\n--- Mamdani (15 regul, baseline) ---")

t_train_start = time.time()
mamdani_bl = build_mamdani_baseline()
t_mamdani_train = time.time() - t_train_start  # "trening" = budowa systemu

n_inference_mamdani = min(200, len(test_df))
t_inf_start = time.time()
for i in range(n_inference_mamdani):
    idx = test_start_idx + i
    window = all_prices[max(0, idx - LOOKBACK):idx + 1]
    if len(window) >= 3:
        zmiana, trend_v, zmiennosc_v = extract_features_baseline(window)
        mamdani_bl.input["zmiana"] = zmiana
        mamdani_bl.input["trend"] = trend_v
        mamdani_bl.input["zmiennosc"] = zmiennosc_v
        try:
            mamdani_bl.compute()
            _ = mamdani_bl.output["prognoza"]
        except Exception:
            pass
t_mamdani_inf = time.time() - t_inf_start
t_mamdani_inf_per_sample = t_mamdani_inf / n_inference_mamdani

mamdani_memory = sys.getsizeof(mamdani_bl) / (1024 ** 2)

RESULTS["Mamdani"] = {
    "czas_treningu_s": round(t_mamdani_train, 4),
    "czas_inferencji_s": round(t_mamdani_inf, 4),
    "czas_inferencji_na_probke_s": round(t_mamdani_inf_per_sample, 6),
    "n_probek_inferencji": n_inference_mamdani,
    "parametry_total": "N/A (15 regul, system ekspertowy)",
    "parametry_trenowalnych": "N/A (brak treningu — reguly eksperckie)",
    "pamiec_modelu_MB": round(mamdani_memory, 4),
    "liczba_regul": 15,
}
print(f"  Budowa: {t_mamdani_train:.4f}s | Inferencja ({n_inference_mamdani} probek): {t_mamdani_inf:.4f}s")

# === 5. TCN+Mamdani (hybrid) ===
print("\n--- TCN+Mamdani (hybrid) ---")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

tcn_fe = TCNFeatureExtractor()
fc_proxy = nn.Linear(3, 1)
total_p_tcn, train_p_tcn = count_parameters(tcn_fe)
total_p_proxy, _ = count_parameters(fc_proxy)
hybrid_total_p = total_p_tcn + total_p_proxy
hybrid_memory = get_model_memory_mb(tcn_fe) + get_model_memory_mb(fc_proxy)

# Trening TCN feature extractor
params = list(tcn_fe.parameters()) + list(fc_proxy.parameters())
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
criterion = nn.MSELoss()
best_val_loss = float("inf")
patience, patience_count = 15, 0
best_state = None

t_train_start = time.time()
for epoch in range(EPOCHS):
    tcn_fe.train()
    fc_proxy.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        features = tcn_fe(xb)
        pred = fc_proxy(features).squeeze()
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    tcn_fe.eval()
    fc_proxy.eval()
    with torch.no_grad():
        val_feat = tcn_fe(X_val)
        val_pred = fc_proxy(val_feat).squeeze()
        val_loss = criterion(val_pred, y_val).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = tcn_fe.state_dict().copy()
        patience_count = 0
    else:
        patience_count += 1
        if patience_count >= patience:
            break

t_hybrid_train = time.time() - t_train_start

tcn_fe.load_state_dict(best_state)
tcn_fe.eval()

# Inferencja: TCN ekstrakcja + Mamdani wnioskowanie
mamdani_sim = build_mamdani()

t_inf_start = time.time()
with torch.no_grad():
    test_features = tcn_fe(X_test).numpy()

for i in range(len(test_features)):
    _ = mamdani_predict(mamdani_sim, test_features[i])
t_hybrid_inf = time.time() - t_inf_start

RESULTS["TCN_Mamdani"] = {
    "czas_treningu_s": round(t_hybrid_train, 4),
    "czas_inferencji_s": round(t_hybrid_inf, 4),
    "czas_inferencji_na_probke_s": round(t_hybrid_inf / len(X_test), 6),
    "n_probek_inferencji": len(X_test),
    "parametry_total": hybrid_total_p,
    "parametry_trenowalnych": train_p_tcn + total_p_proxy,
    "parametry_tcn": total_p_tcn,
    "parametry_proxy_fc": total_p_proxy,
    "pamiec_modelu_MB": round(hybrid_memory, 4),
    "liczba_regul_mamdani": 15,
}
print(f"  Trening: {t_hybrid_train:.4f}s | Inferencja: {t_hybrid_inf:.4f}s | Parametry TCN: {total_p_tcn}")

# === PODSUMOWANIE ===
print("\n=== PODSUMOWANIE ZLOZONOSCI OBLICZENIOWEJ ===")
print(f"{'Model':<20} {'Trening(s)':>12} {'Infer.(s)':>12} {'Parametry':>12} {'Pamiec(MB)':>12}")
print("-" * 72)
for name in ["ARIMA", "LSTM", "TCN", "Mamdani", "TCN_Mamdani"]:
    r = RESULTS[name]
    p = str(r["parametry_total"])
    print(f"{name:<20} {r['czas_treningu_s']:>12.4f} {r['czas_inferencji_s']:>12.4f} {p:>12} {r['pamiec_modelu_MB']:>12.4f}")

result_data = {
    "eksperyment": "Porownanie zlozonosci obliczeniowej 5 modeli",
    "dataset": "SP500",
    "modele": RESULTS,
}
save_results(result_data, os.path.join(os.path.dirname(__file__), "wyniki.json"))
print("\n=== EKSPERYMENT 13 ZAKONCZONE ===")
