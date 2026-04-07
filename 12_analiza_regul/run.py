# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 12: Analiza aktywacji regul Mamdaniego w modelu TCN+Mamdani.

Cel: zbadanie interpretowalnosci modelu — ktore reguly aktywuja sie najczesciej,
jakie maja srednie sily aktywacji, i jak rozklad aktywacji wyglada na zbiorze
testowym S&P 500.

Kluczowe dla argumentu interpretowalnosci w artykule.
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
from utils import prepare_data, compute_all_metrics, inverse_transform, save_results
from config import DATASETS, LOOKBACK, EPOCHS, BATCH_SIZE, LEARNING_RATE, RANDOM_SEED

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


# === DEFINICJE REGUL I FUNKCJI PRZYNALEZNOSCI ===

# Definicja regul w postaci tekstowej (do raportowania)
RULE_DESCRIPTIONS = [
    "R01: IF trend=duzy_wzrost AND momentum=duzy_wzrost THEN prognoza=duzy_wzrost",
    "R02: IF trend=wzrost AND momentum=wzrost THEN prognoza=wzrost",
    "R03: IF trend=spadek AND momentum=spadek THEN prognoza=spadek",
    "R04: IF trend=duzy_spadek AND momentum=duzy_spadek THEN prognoza=duzy_spadek",
    "R05: IF trend=wzrost AND momentum=spadek THEN prognoza=neutralny",
    "R06: IF trend=spadek AND momentum=wzrost THEN prognoza=neutralny",
    "R07: IF trend=neutralny AND momentum=neutralny THEN prognoza=neutralny",
    "R08: IF trend=neutralny AND momentum=wzrost THEN prognoza=wzrost",
    "R09: IF trend=neutralny AND momentum=spadek THEN prognoza=spadek",
    "R10: IF trend=duzy_wzrost AND zmiennosc=wysoka THEN prognoza=neutralny",
    "R11: IF trend=duzy_spadek AND zmiennosc=wysoka THEN prognoza=neutralny",
    "R12: IF trend=wzrost AND zmiennosc=niska THEN prognoza=duzy_wzrost",
    "R13: IF trend=spadek AND zmiennosc=niska THEN prognoza=duzy_spadek",
    "R14: IF momentum=duzy_wzrost AND zmiennosc=srednia THEN prognoza=wzrost",
    "R15: IF momentum=duzy_spadek AND zmiennosc=srednia THEN prognoza=spadek",
]

# Definicja regul jako lista: [(antecedent_terms, consequent_term)]
# Kazda regula to tuple: (lista par (zmienna, term), (wyjscie, term))
RULE_SPECS = [
    # R01: trend=duzy_wzrost & momentum=duzy_wzrost -> duzy_wzrost
    ([("trend", "duzy_wzrost"), ("momentum", "duzy_wzrost")], ("prognoza", "duzy_wzrost")),
    # R02: trend=wzrost & momentum=wzrost -> wzrost
    ([("trend", "wzrost"), ("momentum", "wzrost")], ("prognoza", "wzrost")),
    # R03: trend=spadek & momentum=spadek -> spadek
    ([("trend", "spadek"), ("momentum", "spadek")], ("prognoza", "spadek")),
    # R04: trend=duzy_spadek & momentum=duzy_spadek -> duzy_spadek
    ([("trend", "duzy_spadek"), ("momentum", "duzy_spadek")], ("prognoza", "duzy_spadek")),
    # R05: trend=wzrost & momentum=spadek -> neutralny
    ([("trend", "wzrost"), ("momentum", "spadek")], ("prognoza", "neutralny")),
    # R06: trend=spadek & momentum=wzrost -> neutralny
    ([("trend", "spadek"), ("momentum", "wzrost")], ("prognoza", "neutralny")),
    # R07: trend=neutralny & momentum=neutralny -> neutralny
    ([("trend", "neutralny"), ("momentum", "neutralny")], ("prognoza", "neutralny")),
    # R08: trend=neutralny & momentum=wzrost -> wzrost
    ([("trend", "neutralny"), ("momentum", "wzrost")], ("prognoza", "wzrost")),
    # R09: trend=neutralny & momentum=spadek -> spadek
    ([("trend", "neutralny"), ("momentum", "spadek")], ("prognoza", "spadek")),
    # R10: trend=duzy_wzrost & zmiennosc=wysoka -> neutralny
    ([("trend", "duzy_wzrost"), ("zmiennosc", "wysoka")], ("prognoza", "neutralny")),
    # R11: trend=duzy_spadek & zmiennosc=wysoka -> neutralny
    ([("trend", "duzy_spadek"), ("zmiennosc", "wysoka")], ("prognoza", "neutralny")),
    # R12: trend=wzrost & zmiennosc=niska -> duzy_wzrost
    ([("trend", "wzrost"), ("zmiennosc", "niska")], ("prognoza", "duzy_wzrost")),
    # R13: trend=spadek & zmiennosc=niska -> duzy_spadek
    ([("trend", "spadek"), ("zmiennosc", "niska")], ("prognoza", "duzy_spadek")),
    # R14: momentum=duzy_wzrost & zmiennosc=srednia -> wzrost
    ([("momentum", "duzy_wzrost"), ("zmiennosc", "srednia")], ("prognoza", "wzrost")),
    # R15: momentum=duzy_spadek & zmiennosc=srednia -> spadek
    ([("momentum", "duzy_spadek"), ("zmiennosc", "srednia")], ("prognoza", "spadek")),
]


def get_membership_functions():
    """Zwraca slownik funkcji przynaleznosci (identycznych jak w build_mamdani)."""
    universes = {
        "trend": np.linspace(-3, 3, 1000),
        "zmiennosc": np.linspace(0, 5, 1000),
        "momentum": np.linspace(-3, 3, 1000),
        "prognoza": np.linspace(-5, 5, 1000),
    }

    mf_params = {
        "trend": {
            "duzy_spadek": (-2.5, 0.6),
            "spadek": (-1, 0.4),
            "neutralny": (0, 0.35),
            "wzrost": (1, 0.4),
            "duzy_wzrost": (2.5, 0.6),
        },
        "momentum": {
            "duzy_spadek": (-2.5, 0.6),
            "spadek": (-1, 0.4),
            "neutralny": (0, 0.35),
            "wzrost": (1, 0.4),
            "duzy_wzrost": (2.5, 0.6),
        },
        "zmiennosc": {
            "niska": (0, 0.5),
            "srednia": (2, 0.6),
            "wysoka": (4, 0.7),
        },
        "prognoza": {
            "duzy_spadek": (-2.5, 0.6),
            "spadek": (-1, 0.4),
            "neutralny": (0, 0.35),
            "wzrost": (1, 0.4),
            "duzy_wzrost": (2.5, 0.6),
        },
    }

    return universes, mf_params


def compute_membership(value, mean, sigma):
    """Oblicza wartosc funkcji przynaleznosci Gaussa."""
    return np.exp(-0.5 * ((value - mean) / sigma) ** 2)


def compute_rule_firing_strengths(trend_val, vol_val, mom_val, mf_params):
    """
    Oblicza sile aktywacji kazdej reguly (firing strength) manualnie.
    Uzywa operatora AND = min (t-norma minimum, standard Mamdaniego).

    Zwraca liste 15 wartosci (sila aktywacji kazdej reguly).
    """
    # Oblicz przynaleznosci wejsc do kazdego termu
    memberships = {}

    for term, (mean, sigma) in mf_params["trend"].items():
        memberships[("trend", term)] = compute_membership(trend_val, mean, sigma)

    for term, (mean, sigma) in mf_params["zmiennosc"].items():
        memberships[("zmiennosc", term)] = compute_membership(vol_val, mean, sigma)

    for term, (mean, sigma) in mf_params["momentum"].items():
        memberships[("momentum", term)] = compute_membership(mom_val, mean, sigma)

    # Oblicz sile aktywacji kazdej reguly (AND = min)
    firing_strengths = []
    for antecedents, _ in RULE_SPECS:
        ant_values = [memberships[ant] for ant in antecedents]
        firing_strength = min(ant_values)  # t-norma minimum
        firing_strengths.append(firing_strength)

    return firing_strengths


# === GLOWNA CZESC: TRENING I ANALIZA ===

DATASET_KEY = "SP500"
FIRING_THRESHOLD = 0.1  # prrog aktywacji reguly

print(f"=== Eksperyment 12: Analiza regul Mamdaniego ({DATASETS[DATASET_KEY]['name']}) ===")
t0 = time.time()

# Przygotuj dane
data = prepare_data(DATASET_KEY)
X_train = torch.FloatTensor(data["X_train"]).unsqueeze(-1)
y_train = torch.FloatTensor(data["y_train"])
X_val = torch.FloatTensor(data["X_val"]).unsqueeze(-1)
y_val = torch.FloatTensor(data["y_val"])
X_test = torch.FloatTensor(data["X_test"]).unsqueeze(-1)

scaler = data["scaler"]
train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)

# Faza 1: Trening TCN
print("  Faza 1: Trening TCN feature extractor...")
tcn = TCNFeatureExtractor()
fc_proxy = nn.Linear(3, 1)

params = list(tcn.parameters()) + list(fc_proxy.parameters())
optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
criterion = nn.MSELoss()

best_val_loss = float("inf")
patience, patience_count = 15, 0
best_tcn_state = None

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

# Faza 2: Ekstrakcja cech TCN
print("  Faza 2: Ekstrakcja cech TCN z danych testowych...")
with torch.no_grad():
    test_features = tcn(X_test).numpy()

print(f"  Liczba probek testowych: {len(test_features)}")
print(f"  Zakres cech TCN:")
print(f"    trend:     [{test_features[:, 0].min():.3f}, {test_features[:, 0].max():.3f}]")
print(f"    zmiennosc: [{test_features[:, 1].min():.3f}, {test_features[:, 1].max():.3f}]")
print(f"    momentum:  [{test_features[:, 2].min():.3f}, {test_features[:, 2].max():.3f}]")

# Faza 3: Analiza aktywacji regul
print("  Faza 3: Analiza aktywacji regul Mamdaniego...")
_, mf_params = get_membership_functions()

n_rules = len(RULE_SPECS)
n_samples = len(test_features)

# Macierz aktywacji: [n_samples x n_rules]
activation_matrix = np.zeros((n_samples, n_rules))

for i in range(n_samples):
    trend_val = np.clip(test_features[i, 0], -2.9, 2.9)
    vol_val = np.clip(test_features[i, 1], 0.01, 4.9)
    mom_val = np.clip(test_features[i, 2], -2.9, 2.9)

    firing_strengths = compute_rule_firing_strengths(
        trend_val, vol_val, mom_val, mf_params
    )
    activation_matrix[i, :] = firing_strengths

# Statystyki aktywacji
rule_stats = []
for r in range(n_rules):
    strengths = activation_matrix[:, r]
    active_mask = strengths > FIRING_THRESHOLD
    n_active = int(np.sum(active_mask))
    freq_pct = round(n_active / n_samples * 100, 2)
    avg_strength = round(float(np.mean(strengths)), 6)
    avg_strength_when_active = round(float(np.mean(strengths[active_mask])), 6) if n_active > 0 else 0.0
    max_strength = round(float(np.max(strengths)), 6)
    min_strength = round(float(np.min(strengths)), 6)

    rule_stats.append({
        "regula": RULE_DESCRIPTIONS[r],
        "aktywacje_ponad_prog": n_active,
        "czestotliwosc_pct": freq_pct,
        "srednia_sila": avg_strength,
        "srednia_sila_gdy_aktywna": avg_strength_when_active,
        "max_sila": max_strength,
        "min_sila": min_strength,
    })

# Sortuj wg czestotliwosci aktywacji (malejaco)
rule_stats_sorted = sorted(rule_stats, key=lambda x: x["czestotliwosc_pct"], reverse=True)

# Top-5 i bottom-5
top5 = rule_stats_sorted[:5]
bottom5 = rule_stats_sorted[-5:]

print("\n  === TOP-5 NAJCZESCIEJ AKTYWNYCH REGUL ===")
for i, rs in enumerate(top5):
    print(f"  {i+1}. {rs['regula']}")
    print(f"     Czestotliwosc: {rs['czestotliwosc_pct']}% | Srednia sila: {rs['srednia_sila']:.4f} | Max: {rs['max_sila']:.4f}")

print("\n  === BOTTOM-5 NAJRZADZIEJ AKTYWNYCH REGUL ===")
for i, rs in enumerate(bottom5):
    print(f"  {i+1}. {rs['regula']}")
    print(f"     Czestotliwosc: {rs['czestotliwosc_pct']}% | Srednia sila: {rs['srednia_sila']:.4f} | Max: {rs['max_sila']:.4f}")

# Statystyki ogolne
avg_active_rules_per_sample = np.mean(np.sum(activation_matrix > FIRING_THRESHOLD, axis=1))
total_activations = np.sum(activation_matrix > FIRING_THRESHOLD)

elapsed = time.time() - t0

print(f"\n  === STATYSTYKI OGOLNE ===")
print(f"  Srednia liczba aktywnych regul na probke: {avg_active_rules_per_sample:.2f}")
print(f"  Laczna liczba aktywacji (ponad prog {FIRING_THRESHOLD}): {int(total_activations)}")
print(f"  Czas analizy: {round(elapsed, 2)}s")

# Zapisz wyniki
result_data = {
    "eksperyment": "Analiza aktywacji regul Mamdaniego w modelu TCN+Mamdani",
    "dataset": "SP500",
    "prog_aktywacji": FIRING_THRESHOLD,
    "liczba_regul": n_rules,
    "liczba_probek_testowych": n_samples,
    "srednia_aktywnych_regul_na_probke": round(avg_active_rules_per_sample, 2),
    "laczna_liczba_aktywacji": int(total_activations),
    "top5_najczesciej_aktywne": top5,
    "bottom5_najrzadziej_aktywne": bottom5,
    "wszystkie_reguly_statystyki": rule_stats_sorted,
    "czas_s": round(elapsed, 2),
}
save_results(result_data, os.path.join(os.path.dirname(__file__), "wyniki.json"))
print("\n=== EKSPERYMENT 12 ZAKONCZONE ===")
