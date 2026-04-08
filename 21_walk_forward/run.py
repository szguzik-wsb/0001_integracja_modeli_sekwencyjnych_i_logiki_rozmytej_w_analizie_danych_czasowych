# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 21: Walk-forward validation TCN-Mamdani vs ARIMA.

Cel: Wyeliminowac zarzut o jednorazowym podziale danych (70/15/15).
Metoda: 5 rund walk-forward z przesuwanym oknem treningowym.
  - Kazda runda: 60% trening, 10% walidacja, 30% test
  - Okno przesuwa sie o 10% danych miedzy rundami
  - Runda 1: dane[0:60] train, [60:70] val, [70:100] test   (bessa/hossa mix)
  - Runda 2: dane[10:70] train, [70:80] val, [80:100] test  (przesuniete)
  - ...
  - Runda 5: dane[40:100] train (wewn. split)                (najnowsze dane)

Modele: TCN-Mamdani (glowny) vs ARIMA (benchmark).
Zbiory: SP500, WIG20, EURUSD, BTCUSD.

Zrodla:
  - Tashman (2000) — out-of-sample tests of forecasting accuracy
  - Bergmeir & Benitez (2012) — cross-validation for time series
"""
import sys, os, time, warnings, datetime
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA as ARIMA_Model
from utils import load_dataset, compute_all_metrics, save_results
from config import DATASETS, LOOKBACK, EPOCHS, BATCH_SIZE, LEARNING_RATE, RANDOM_SEED

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================================
# PARAMETRY WALK-FORWARD
# ============================================================================
N_FOLDS = 5
TRAIN_PCT = 0.60   # 60% trening
VAL_PCT = 0.10     # 10% walidacja
TEST_PCT = 0.30    # 30% test
SHIFT_PCT = 0.10   # przesuniecie miedzy rundami

DATASET_KEYS = ["SP500", "WIG20", "EURUSD", "BTCUSD"]


# ============================================================================
# TCN FEATURE EXTRACTOR (identyczny z exp 05)
# ============================================================================
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


# ============================================================================
# MAMDANI (identyczny z exp 05)
# ============================================================================
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


# ============================================================================
# HELPER: create sequences
# ============================================================================
def create_sequences(data, lookback=LOOKBACK):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)


# ============================================================================
# TCN-MAMDANI: train + predict for one fold
# ============================================================================
def run_tcn_mamdani_fold(train_prices, val_prices, test_prices, fold_num):
    """Trenuje TCN-Mamdani na jednej rundzie walk-forward."""
    scaler = MinMaxScaler()
    train_s = scaler.fit_transform(train_prices.reshape(-1, 1)).flatten()
    val_s = scaler.transform(val_prices.reshape(-1, 1)).flatten()
    test_s = scaler.transform(test_prices.reshape(-1, 1)).flatten()

    X_train, y_train = create_sequences(train_s)
    X_val, y_val = create_sequences(val_s)
    X_test, y_test = create_sequences(test_s)

    if len(X_train) < 10 or len(X_test) < 5:
        return None

    X_tr = torch.FloatTensor(X_train).unsqueeze(-1)
    y_tr = torch.FloatTensor(y_train)
    X_v = torch.FloatTensor(X_val).unsqueeze(-1)
    y_v = torch.FloatTensor(y_val)
    X_te = torch.FloatTensor(X_test).unsqueeze(-1)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    tcn = TCNFeatureExtractor()
    proxy = nn.Linear(3, 1)
    optimizer = torch.optim.Adam(list(tcn.parameters()) + list(proxy.parameters()), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val = float("inf")
    patience_count = 0

    for epoch in range(EPOCHS):
        tcn.train(); proxy.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = proxy(tcn(xb)).squeeze()
            criterion(pred, yb).backward()
            optimizer.step()

        tcn.eval(); proxy.eval()
        with torch.no_grad():
            vl = criterion(proxy(tcn(X_v)).squeeze(), y_v).item()
        if vl < best_val:
            best_val = vl
            best_state = tcn.state_dict().copy()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 15:
                break

    tcn.load_state_dict(best_state)
    tcn.eval()

    sim = build_mamdani()
    with torch.no_grad():
        feats = tcn(X_te).numpy()

    preds_s = []
    for i in range(len(feats)):
        pc = mamdani_predict(sim, feats[i])
        preds_s.append(X_test[i, -1] * (1 + pc / 100))

    preds = scaler.inverse_transform(np.array(preds_s).reshape(-1, 1)).flatten()
    actuals = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    return compute_all_metrics(actuals, preds)


# ============================================================================
# ARIMA: predict for one fold
# ============================================================================
def run_arima_fold(train_prices, test_prices, fold_num):
    """ARIMA(5,1,0) walk-forward na jednej rundzie."""
    history = list(train_prices)
    preds = []

    for i in range(len(test_prices)):
        try:
            model = ARIMA_Model(history, order=(5, 1, 0))
            fit = model.fit()
            pred = fit.forecast(steps=1)[0]
        except Exception:
            pred = history[-1]
        preds.append(pred)
        history.append(test_prices[i])

    return compute_all_metrics(test_prices, np.array(preds))


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("  EKSPERYMENT 21: WALK-FORWARD VALIDATION")
    print("  TCN-Mamdani vs ARIMA | 5 rund | 4 zbiory danych")
    print("=" * 70, flush=True)

    t0_total = time.time()
    all_results = {}

    for ds_key in DATASET_KEYS:
        ds_name = DATASETS[ds_key]["name"]
        print(f"\n{'='*60}")
        print(f"  ZBIOR: {ds_name}")
        print(f"{'='*60}", flush=True)

        df = load_dataset(ds_key)
        prices = df["Close"].values
        n = len(prices)

        # Expanding window: staly test_size, trening rosnie
        test_size = int(n * 0.15)  # ~15% danych na test w kazdym foldzie
        val_size = int(n * 0.05)   # ~5% na walidacje
        min_train = int(n * 0.40)  # min 40% na trening

        # Oblicz punkty podzialu dla 5 foldow
        # Fold 1: train=[0:40%], val=[40:45%], test=[45:60%]
        # Fold 2: train=[0:55%], val=[55:60%], test=[60:75%]
        # ...
        # Fold 5: train=[0:80%], val=[80:85%], test=[85:100%]
        step = (n - min_train - val_size - test_size) // (N_FOLDS - 1) if N_FOLDS > 1 else 0

        ds_results = {"folds": [], "hybrid_mapes": [], "arima_mapes": []}

        for fold in range(N_FOLDS):
            now = datetime.datetime.now().strftime("%H:%M:%S")
            train_end = min_train + fold * step
            val_end = train_end + val_size
            test_end = min(val_end + test_size, n)

            if train_end >= n or val_end >= n or test_end - val_end < LOOKBACK + 5:
                print(f"  [{now}] Fold {fold+1}/{N_FOLDS}: POMINIETY (za malo danych)", flush=True)
                continue

            train_p = prices[0:train_end]
            val_p = prices[train_end:val_end]
            test_p = prices[val_end:test_end]

            period_start = df["Date"].iloc[0]
            period_test_start = df["Date"].iloc[val_end]
            period_end = df["Date"].iloc[min(test_end - 1, n - 1)]

            print(f"  [{now}] Fold {fold+1}/{N_FOLDS}: "
                  f"train={len(train_p)}, val={len(val_p)}, test={len(test_p)} "
                  f"(test: {str(period_test_start)[:10]} -> {str(period_end)[:10]})", flush=True)

            # TCN-Mamdani
            print(f"    TCN-Mamdani: trening...", end="", flush=True)
            t1 = time.time()
            hybrid_metrics = run_tcn_mamdani_fold(train_p, val_p, test_p, fold)
            t_hybrid = time.time() - t1
            if hybrid_metrics:
                print(f" MAPE={hybrid_metrics['MAPE']:.4f}% ({t_hybrid:.0f}s)", flush=True)
            else:
                print(f" BLAD", flush=True)
                continue

            # ARIMA
            print(f"    ARIMA:       trening...", end="", flush=True)
            t2 = time.time()
            arima_metrics = run_arima_fold(train_p, test_p, fold)
            t_arima = time.time() - t2
            print(f" MAPE={arima_metrics['MAPE']:.4f}% ({t_arima:.0f}s)", flush=True)

            fold_result = {
                "fold": fold + 1,
                "train_size": len(train_p),
                "test_size": len(test_p),
                "period": f"{str(period_start)[:10]} - {str(period_end)[:10]}",
                "hybrid": hybrid_metrics,
                "arima": arima_metrics,
            }
            ds_results["folds"].append(fold_result)
            ds_results["hybrid_mapes"].append(hybrid_metrics["MAPE"])
            ds_results["arima_mapes"].append(arima_metrics["MAPE"])

        # Podsumowanie zbioru
        if ds_results["hybrid_mapes"]:
            h_mean = np.mean(ds_results["hybrid_mapes"])
            h_std = np.std(ds_results["hybrid_mapes"])
            a_mean = np.mean(ds_results["arima_mapes"])
            a_std = np.std(ds_results["arima_mapes"])
            wins = sum(1 for h, a in zip(ds_results["hybrid_mapes"], ds_results["arima_mapes"]) if h < a)

            ds_results["summary"] = {
                "hybrid_MAPE_mean": round(h_mean, 4),
                "hybrid_MAPE_std": round(h_std, 4),
                "arima_MAPE_mean": round(a_mean, 4),
                "arima_MAPE_std": round(a_std, 4),
                "hybrid_wins": wins,
                "total_folds": len(ds_results["hybrid_mapes"]),
            }

            print(f"\n  --- PODSUMOWANIE {ds_name} ---")
            print(f"  TCN-Mamdani: MAPE = {h_mean:.4f} +/- {h_std:.4f}%")
            print(f"  ARIMA:       MAPE = {a_mean:.4f} +/- {a_std:.4f}%")
            print(f"  Hybrid wygrywa: {wins}/{len(ds_results['hybrid_mapes'])} rund", flush=True)

        all_results[ds_key] = ds_results

    elapsed = time.time() - t0_total
    all_results["czas_calkowity_s"] = round(elapsed, 1)

    save_results(all_results, os.path.join(os.path.dirname(__file__), "wyniki.json"))

    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD ZAKONCZONY  |  Czas: {elapsed/60:.1f} min")
    print(f"{'='*70}", flush=True)


if __name__ == "__main__":
    main()
