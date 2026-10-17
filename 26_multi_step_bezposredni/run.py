# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 26: Prognozy wielokrokowe TCN-Mamdani w strategii bezposredniej (direct)
na S&P 500 — uzupelnienie eksperymentu 08 (tam: strategia rekurencyjna).

Strategia bezposrednia: dla kazdego horyzontu h osobny ekstraktor TCN trenowany z warstwa
proxy regresujaca znormalizowana cene z dnia t+h (ta sama architektura, optymalizator,
early stopping i ziarno co w 05/08); warstwa Mamdaniego (15 regul, parametry z 05)
wnioskuje JEDEN raz, a jej wyjscie y* traktowane jest jako procentowa zmiana do horyzontu h:
P_hat[t+h] = P[t] * (1 + y*/100) w skali znormalizowanej, nastepnie odwrocenie skali.
Ograniczenie: dziedzina wyjscia Mamdaniego [-5, 5] ogranicza prognozowana zmiane.
Metryka jak w 08: MAPE dla ostatniego kroku horyzontu, te same punkty startowe.
Kontekst: prognoza naiwna P_hat[t+h] = P[t] na tych samych punktach oraz diagnostyka
rekurencji — rozklad jednokrokowej zmiany y* modelu z 05 (czy stale dodatnia).
"""
import sys, os, time, json, warnings
import copy
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from utils import load_dataset, split_data, save_results
from config import DATASETS, LOOKBACK, EPOCHS, BATCH_SIZE, LEARNING_RATE, RANDOM_SEED

HORIZONS = [1, 5, 10, 30]
DATASET_KEY = "SP500"
OUT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(OUT)


def prepare_data_multistep(dataset_key, lookback=LOOKBACK, horizon=1):
    """Pipeline danych z konfigurowalnym horyzontem (identyczny jak w 08)."""
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
    return {"X_train": X_train, "y_train": y_train, "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test, "scaler": scaler}


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


def build_mamdani():
    """System Mamdaniego z 05 (15 regul, parametry reczne)."""
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
    return ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))


def mamdani_predict(sim, tcn_features):
    sim.input["trend"] = np.clip(tcn_features[0], -2.9, 2.9)
    sim.input["zmiennosc"] = np.clip(tcn_features[1], 0.01, 4.9)
    sim.input["momentum"] = np.clip(tcn_features[2], -2.9, 2.9)
    try:
        sim.compute()
        return sim.output["prognoza"]
    except Exception:
        return 0.0


def run_direct(horizon):
    t0 = time.time()
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    data = prepare_data_multistep(DATASET_KEY, LOOKBACK, horizon)
    X_train = torch.FloatTensor(data["X_train"]).unsqueeze(-1)
    y_train = torch.FloatTensor(data["y_train"][:, -1])
    X_val = torch.FloatTensor(data["X_val"]).unsqueeze(-1)
    y_val = torch.FloatTensor(data["y_val"][:, -1])
    X_test = torch.FloatTensor(data["X_test"]).unsqueeze(-1)
    scaler = data["scaler"]
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    tcn = TCNFeatureExtractor()
    fc_proxy = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(list(tcn.parameters()) + list(fc_proxy.parameters()), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_val, patience_count = float("inf"), 0
    for epoch in range(EPOCHS):
        tcn.train(); fc_proxy.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(fc_proxy(tcn(xb)).squeeze(), yb)
            loss.backward()
            optimizer.step()
        tcn.eval(); fc_proxy.eval()
        with torch.no_grad():
            val_loss = criterion(fc_proxy(tcn(X_val)).squeeze(), y_val).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(tcn.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 15:
                break
    tcn.load_state_dict(best_state)
    tcn.eval()

    sim = build_mamdani()
    with torch.no_grad():
        feats = tcn(X_test).numpy()
    change = np.array([mamdani_predict(sim, f) for f in feats])
    last = data["X_test"][:, -1]
    pred = scaler.inverse_transform((last * (1 + change / 100)).reshape(-1, 1)).flatten()
    actual = scaler.inverse_transform(data["y_test"][:, -1].reshape(-1, 1)).flatten()
    origin = scaler.inverse_transform(last.reshape(-1, 1)).flatten()
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    mape_naive = float(np.mean(np.abs((actual - origin) / actual)) * 100)
    res = {
        "MAPE_TCN_Mamdani_bezposredni": round(mape, 4),
        "MAPE_naiwna_P_t": round(mape_naive, 4),
        "n_punktow": int(len(actual)),
        "epoki_treningu": epoch + 1,
        "odsetek_prognoz_wzrostu": round(float(np.mean(pred > origin)), 3),
        "srednia_prognozowana_zmiana_pct_ceny": round(float(np.mean((pred / origin - 1) * 100)), 4),
        "srednia_rzeczywista_zmiana_pct_ceny": round(float(np.mean((actual / origin - 1) * 100)), 4),
        "czas_s": round(time.time() - t0, 2),
    }
    print(f"h={horizon:2d}: direct={res['MAPE_TCN_Mamdani_bezposredni']}% naiwna={res['MAPE_naiwna_P_t']}% "
          f"wzrost={res['odsetek_prognoz_wzrostu']} zm_prog={res['srednia_prognozowana_zmiana_pct_ceny']}% "
          f"zm_rzecz={res['srednia_rzeczywista_zmiana_pct_ceny']}% n={res['n_punktow']}")
    return res


# Diagnostyka rekurencji: jednokrokowa zmiana y* modelu z 05 na S&P 500 (w skali znormalizowanej)
data1 = prepare_data_multistep(DATASET_KEY, LOOKBACK, 1)
p05 = pd.read_csv(os.path.join(BASE, "05_tcn_mamdani_hybrid", f"prognozy_{DATASET_KEY}.csv"))
pred_scaled = data1["scaler"].transform(p05[["Predicted"]].values).flatten()
last_scaled = data1["X_test"][:, -1]
ystar = (pred_scaled / last_scaled - 1) * 100
diag = {
    "y_star_srednia_pct_skali_znorm": round(float(ystar.mean()), 4),
    "y_star_min": round(float(ystar.min()), 4),
    "y_star_max": round(float(ystar.max()), 4),
    "odsetek_y_star_dodatnich": round(float(np.mean(ystar > 0)), 3),
    "srednia_jednokrokowa_zmiana_pct_ceny": round(float(np.mean((p05["Predicted"].values /
        data1["scaler"].inverse_transform(last_scaled.reshape(-1, 1)).flatten() - 1) * 100)), 4),
}
print("Diagnostyka rekurencji (05):", diag)

with open(os.path.join(BASE, "08_multi_step", "wyniki.json"), encoding="utf-8") as f:
    rec = json.load(f)["modele"]

RESULTS = {}
for h in HORIZONS:
    r = run_direct(h)
    r["MAPE_TCN_Mamdani_rekurencyjny_08"] = rec["TCN_Mamdani"][f"h={h}"]["MAPE"]
    r["MAPE_ARIMA_08"] = rec["ARIMA"][f"h={h}"]["MAPE"]
    r["MAPE_LSTM_bezposredni_08"] = rec["LSTM"][f"h={h}"]["MAPE"]
    RESULTS[f"h={h}"] = r

save_results({
    "eksperyment": "Prognozy wielokrokowe TCN-Mamdani w strategii bezposredniej (S&P 500)",
    "metryka": "MAPE dla ostatniego kroku horyzontu (jak w 08)",
    "diagnostyka_rekurencji_05": diag,
    "wyniki": RESULTS,
}, os.path.join(OUT, "wyniki.json"))
print("\n=== EKSPERYMENT 26 ZAKONCZONY ===")
