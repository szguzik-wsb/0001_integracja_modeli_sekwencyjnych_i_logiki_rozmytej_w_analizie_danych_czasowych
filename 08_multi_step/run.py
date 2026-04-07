"""
Eksperyment 08: Prognozowanie wielokrokowe (multi-step forecasting).

Porownanie TCN+Mamdani vs ARIMA vs LSTM dla roznych horyzontow prognozy:
h = 1, 5, 10, 30 dni na zbiorze S&P 500.

Strategie:
  - ARIMA: forecast(steps=h) — natywna prognoza wielokrokowa
  - LSTM: osobny model trenowany dla kazdego horyzontu h
  - TCN+Mamdani: osobny model trenowany dla kazdego horyzontu h,
    TCN ekstrahuje cechy, Mamdani wnioskuje prognoze srednia (mean)
    na h krokow naprzod

Metryka glowna: MAPE dla ostatniego kroku horyzontu.
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from utils import load_dataset, split_data, compute_all_metrics, save_results
from config import DATASETS, LOOKBACK, EPOCHS, BATCH_SIZE, LEARNING_RATE, RANDOM_SEED

HORIZONS = [1, 5, 10, 30]
DATASET_KEY = "SP500"
EXP_DIR = os.path.dirname(os.path.abspath(__file__))


# === PRZYGOTOWANIE DANYCH DLA ROZNYCH HORYZONTOW ===

def prepare_data_multistep(dataset_key, lookback=LOOKBACK, horizon=1):
    """Pipeline danych z konfigurowalnym horyzontem."""
    df = load_dataset(dataset_key)
    train_df, val_df, test_df = split_data(df)

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df[["Close"]].values).flatten()
    val_scaled = scaler.transform(val_df[["Close"]].values).flatten()
    test_scaled = scaler.transform(test_df[["Close"]].values).flatten()

    def create_seq(data, lb, h):
        X, y = [], []
        for i in range(len(data) - lb - h + 1):
            X.append(data[i:i + lb])
            y.append(data[i + lb:i + lb + h])
        return np.array(X), np.array(y)

    X_train, y_train = create_seq(train_scaled, lookback, horizon)
    X_val, y_val = create_seq(val_scaled, lookback, horizon)
    X_test, y_test = create_seq(test_scaled, lookback, horizon)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
    }


# === LSTM MODEL ===

class LSTMMultiStep(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 dropout=0.2, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


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


# === ARIMA MULTI-STEP ===

def run_arima_multistep(horizon):
    """ARIMA(5,1,0) z natywna prognoza wielokrokowa forecast(steps=h)."""
    print(f"\n  ARIMA h={horizon}")
    t0 = time.time()

    df = load_dataset(DATASET_KEY)
    train_df, val_df, test_df = split_data(df)

    history = list(train_df["Close"].values)
    history.extend(list(val_df["Close"].values))
    test_vals = test_df["Close"].values

    # Prognoza: dla kazdego punktu startowego prognozujemy h krokow
    all_mape = []
    n_steps = len(test_vals) - horizon + 1

    for i in range(n_steps):
        try:
            model = ARIMA(history, order=(5, 1, 0))
            fitted = model.fit()
            yhat = fitted.forecast(steps=horizon)
            actual_h = test_vals[i:i + horizon]

            # MAPE na ostatnim kroku horyzontu
            mape_val = np.abs((actual_h[-1] - yhat[-1]) / actual_h[-1]) * 100
            all_mape.append(mape_val)
        except Exception:
            pass

        history.append(test_vals[i])

        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n_steps}")

    elapsed = time.time() - t0
    avg_mape = round(float(np.mean(all_mape)), 4)
    print(f"    MAPE: {avg_mape}%, czas: {round(elapsed, 2)}s")
    return {"MAPE": avg_mape, "czas_s": round(elapsed, 2)}


# === LSTM MULTI-STEP ===

def run_lstm_multistep(horizon):
    """LSTM z bezposrednia prognoza h-krokowa (direct strategy)."""
    print(f"\n  LSTM h={horizon}")
    t0 = time.time()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    data = prepare_data_multistep(DATASET_KEY, LOOKBACK, horizon)
    X_train = torch.FloatTensor(data["X_train"]).unsqueeze(-1)
    y_train = torch.FloatTensor(data["y_train"])
    X_val = torch.FloatTensor(data["X_val"]).unsqueeze(-1)
    y_val = torch.FloatTensor(data["y_val"])
    X_test = torch.FloatTensor(data["X_test"]).unsqueeze(-1)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMMultiStep(horizon=horizon)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience, patience_count = 15, 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    scaler = data["scaler"]
    with torch.no_grad():
        preds_scaled = model(X_test).numpy()

    # MAPE na ostatnim kroku horyzontu
    y_test = data["y_test"]
    if horizon == 1:
        pred_last = preds_scaled.flatten()
        actual_last = y_test.flatten()
    else:
        pred_last = preds_scaled[:, -1]
        actual_last = y_test[:, -1]

    pred_inv = scaler.inverse_transform(pred_last.reshape(-1, 1)).flatten()
    actual_inv = scaler.inverse_transform(actual_last.reshape(-1, 1)).flatten()

    mask = actual_inv != 0
    avg_mape = round(float(np.mean(np.abs(
        (actual_inv[mask] - pred_inv[mask]) / actual_inv[mask]
    )) * 100), 4)

    elapsed = time.time() - t0
    print(f"    MAPE: {avg_mape}%, czas: {round(elapsed, 2)}s")
    return {"MAPE": avg_mape, "czas_s": round(elapsed, 2)}


# === TCN + MAMDANI MULTI-STEP ===

def run_tcn_mamdani_multistep(horizon):
    """
    TCN+Mamdani z rekurencyjna prognoza wielokrokowa.
    Dla h>1: predykcja jest powtarzana iteracyjnie h razy,
    przy czym kolejna predykcja korzysta z poprzednich wynikow.
    """
    print(f"\n  TCN+Mamdani h={horizon}")
    t0 = time.time()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Dane dla h=1 (model trenowany na jednym kroku)
    data = prepare_data_multistep(DATASET_KEY, LOOKBACK, horizon=1)
    X_train = torch.FloatTensor(data["X_train"]).unsqueeze(-1)
    y_train = torch.FloatTensor(data["y_train"])
    X_val = torch.FloatTensor(data["X_val"]).unsqueeze(-1)
    y_val = torch.FloatTensor(data["y_val"])

    scaler = data["scaler"]
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)

    # Trening TCN feature extractor
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
            loss = criterion(pred, yb.squeeze())
            loss.backward()
            optimizer.step()

        tcn.eval(); fc_proxy.eval()
        with torch.no_grad():
            val_feat = tcn(X_val)
            val_pred = fc_proxy(val_feat).squeeze()
            val_loss = criterion(val_pred, y_val.squeeze()).item()

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

    mamdani_sim = build_mamdani()

    # Rekurencyjna prognoza wielokrokowa
    test_scaled = scaler.transform(
        data["test_df"][["Close"]].values
    ).flatten()

    all_mape = []
    n_steps = len(test_scaled) - LOOKBACK - horizon + 1

    for i in range(n_steps):
        window = test_scaled[i:i + LOOKBACK].copy()
        preds_h = []

        for step in range(horizon):
            x_input = torch.FloatTensor(window[-LOOKBACK:]).unsqueeze(0).unsqueeze(-1)
            with torch.no_grad():
                feat = tcn(x_input).numpy()[0]

            pred_change = mamdani_predict(mamdani_sim, feat)
            pred_val = window[-1] * (1 + pred_change / 100)
            preds_h.append(pred_val)
            window = np.append(window, pred_val)

        # MAPE na ostatnim kroku
        actual_val = test_scaled[i + LOOKBACK + horizon - 1]
        actual_inv = scaler.inverse_transform([[actual_val]])[0, 0]
        pred_inv = scaler.inverse_transform([[preds_h[-1]]])[0, 0]

        if actual_inv != 0:
            mape_val = np.abs((actual_inv - pred_inv) / actual_inv) * 100
            all_mape.append(mape_val)

    elapsed = time.time() - t0
    avg_mape = round(float(np.mean(all_mape)), 4)
    print(f"    MAPE: {avg_mape}%, czas: {round(elapsed, 2)}s")
    return {"MAPE": avg_mape, "czas_s": round(elapsed, 2)}


# === GLOWNA PETLA ===

print("=" * 60)
print("EKSPERYMENT 08: PROGNOZOWANIE WIELOKROKOWE (MULTI-STEP)")
print(f"Zbior danych: {DATASETS[DATASET_KEY]['name']}")
print(f"Horyzonty: {HORIZONS}")
print("=" * 60)

results = {
    "opis": "Prognozowanie wielokrokowe: MAPE na ostatnim kroku horyzontu",
    "zbior_danych": DATASET_KEY,
    "horyzonty": HORIZONS,
    "modele": {}
}

# ARIMA
print("\n--- ARIMA(5,1,0) ---")
arima_results = {}
for h in HORIZONS:
    arima_results[f"h={h}"] = run_arima_multistep(h)
results["modele"]["ARIMA"] = arima_results

# LSTM
print("\n--- LSTM(hidden=64, layers=2) ---")
lstm_results = {}
for h in HORIZONS:
    lstm_results[f"h={h}"] = run_lstm_multistep(h)
results["modele"]["LSTM"] = lstm_results

# TCN + Mamdani
print("\n--- TCN+Mamdani ---")
tcn_mamdani_results = {}
for h in HORIZONS:
    tcn_mamdani_results[f"h={h}"] = run_tcn_mamdani_multistep(h)
results["modele"]["TCN_Mamdani"] = tcn_mamdani_results

# Podsumowanie
print("\n" + "=" * 60)
print("PODSUMOWANIE MAPE [%] wg horyzontu:")
print(f"{'Horyzont':<12} {'ARIMA':<12} {'LSTM':<12} {'TCN+Mamdani':<12}")
print("-" * 48)
for h in HORIZONS:
    key = f"h={h}"
    a = arima_results[key]["MAPE"]
    l = lstm_results[key]["MAPE"]
    t = tcn_mamdani_results[key]["MAPE"]
    print(f"h={h:<9} {a:<12} {l:<12} {t:<12}")

save_results(results, os.path.join(EXP_DIR, "wyniki.json"))
print("\n=== EKSPERYMENT 08 ZAKONCZONY ===")
