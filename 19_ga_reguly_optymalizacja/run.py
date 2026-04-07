# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 19: Optymalizacja bazy regul Mamdaniego algorytmem genetycznym.

Punkt wyjscia: 15 regul recznie zdefiniowanych z eksperymentu 05.
GA optymalizuje:
  (a) parametry funkcji przynaleznosci (centra i szerokosci Gaussowskich MF)
  (b) selekcje regul (ktore z 15 regul zachowac)

Fitness = -MAPE na zbiorze walidacyjnym.

GA: populacja=30, generacje=50, krzyzowanie=0.8, mutacja=0.1,
    selekcja turniejowa (k=3), krzyzowanie jednolite, mutacja gaussowska.

Uruchamiany tylko na S&P 500 (porownanie: recznie vs GA-optymalizowane).

Zrodla:
  - Holland (1975) — algorytmy genetyczne
  - Cordon (2011) — przeglad ewolucyjnego uczenia systemow Mamdaniego
  - Alves & Aguiar (2025) — GEN-NMR, genetyczna optymalizacja regresora Mamdaniego
"""
import sys, os, time, warnings, datetime

# Wylacz buforowanie stdout/stderr GLOBALNIE
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from utils import prepare_data, compute_all_metrics, inverse_transform, save_results
from config import DATASETS, LOOKBACK, RANDOM_SEED

np.random.seed(RANDOM_SEED)


# ============================================================================
# DEFINICJA CHROMOSOMU
# ============================================================================
# Zmienne lingwistyczne i ich termy:
#   trend:     duzy_spadek, spadek, neutralny, wzrost, duzy_wzrost  (5 x 2 = 10 params)
#   zmiennosc: niska, srednia, wysoka                               (3 x 2 =  6 params)
#   momentum:  duzy_spadek, spadek, neutralny, wzrost, duzy_wzrost  (5 x 2 = 10 params)
#   prognoza:  duzy_spadek, spadek, neutralny, wzrost, duzy_wzrost  (5 x 2 = 10 params)
# Razem parametry MF: 36
# Selekcja regul: 15 bitow (1=aktywna, 0=wylaczona)
# Lacznie chromosom: 36 (float) + 15 (float, progowane na 0.5)

NUM_MF_PARAMS = 36
NUM_RULES = 15
CHROM_LEN = NUM_MF_PARAMS + NUM_RULES

# Domyslne wartosci MF z eksperymentu 05 (centra i szerokosci)
DEFAULT_MF_PARAMS = np.array([
    # trend (5 termow): [center, width] x 5
    -2.5, 0.6,   # duzy_spadek
    -1.0, 0.4,   # spadek
     0.0, 0.35,  # neutralny
     1.0, 0.4,   # wzrost
     2.5, 0.6,   # duzy_wzrost
    # zmiennosc (3 termy): [center, width] x 3
     0.0, 0.5,   # niska
     2.0, 0.6,   # srednia
     4.0, 0.7,   # wysoka
    # momentum (5 termow): [center, width] x 5
    -2.5, 0.6,   # duzy_spadek
    -1.0, 0.4,   # spadek
     0.0, 0.35,  # neutralny
     1.0, 0.4,   # wzrost
     2.5, 0.6,   # duzy_wzrost
    # prognoza (5 termow): [center, width] x 5
    -2.5, 0.6,   # duzy_spadek
    -1.0, 0.4,   # spadek
     0.0, 0.35,  # neutralny
     1.0, 0.4,   # wzrost
     2.5, 0.6,   # duzy_wzrost
], dtype=np.float64)

# Domyslna selekcja regul: wszystkie aktywne
DEFAULT_RULE_SELECTION = np.ones(NUM_RULES, dtype=np.float64)


def decode_chromosome(chrom):
    """Dekoduj chromosom na parametry MF i selekcje regul."""
    mf_params = chrom[:NUM_MF_PARAMS]
    rule_bits = chrom[NUM_MF_PARAMS:]
    rule_active = (rule_bits >= 0.5).astype(bool)
    return mf_params, rule_active


def build_mamdani_from_params(mf_params, rule_active):
    """Buduje system Mamdaniego z podanych parametrow MF i aktywnych regul."""
    # Rozpakuj parametry MF
    idx = 0

    def next_mf():
        nonlocal idx
        center, width = mf_params[idx], max(mf_params[idx + 1], 0.1)
        idx += 2
        return center, width

    # Zmienne lingwistyczne
    trend = ctrl.Antecedent(np.linspace(-3, 3, 1000), "trend")
    zmiennosc = ctrl.Antecedent(np.linspace(0, 5, 1000), "zmiennosc")
    momentum = ctrl.Antecedent(np.linspace(-3, 3, 1000), "momentum")
    prognoza = ctrl.Consequent(np.linspace(-5, 5, 1000), "prognoza")

    # Trend MF
    term_names_5 = ["duzy_spadek", "spadek", "neutralny", "wzrost", "duzy_wzrost"]
    for name in term_names_5:
        c, w = next_mf()
        trend[name] = fuzz.gaussmf(trend.universe, c, w)

    # Zmiennosc MF
    term_names_3 = ["niska", "srednia", "wysoka"]
    for name in term_names_3:
        c, w = next_mf()
        zmiennosc[name] = fuzz.gaussmf(zmiennosc.universe, c, w)

    # Momentum MF
    for name in term_names_5:
        c, w = next_mf()
        momentum[name] = fuzz.gaussmf(momentum.universe, c, w)

    # Prognoza MF
    for name in term_names_5:
        c, w = next_mf()
        prognoza[name] = fuzz.gaussmf(prognoza.universe, c, w)

    # Definicja wszystkich 15 regul (identyczne z eksperymentem 05)
    all_rules = [
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

    # Filtruj aktywne reguly
    active_rules = [r for r, active in zip(all_rules, rule_active) if active]

    if len(active_rules) < 2:
        # Minimum 2 reguly potrzebne do dzialania systemu
        return None

    try:
        system = ctrl.ControlSystem(active_rules)
        sim = ctrl.ControlSystemSimulation(system)
        return sim
    except Exception:
        return None


def mamdani_predict(sim, tcn_features):
    """Prognoza Mamdaniego na podstawie cech (trend, zmiennosc, momentum)."""
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


# ============================================================================
# TCN FEATURE EXTRACTOR (z eksperymentu 05, wytrenowany przed GA)
# ============================================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE

torch.manual_seed(RANDOM_SEED)


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


def train_tcn_extractor(data):
    """Trenuje TCN feature extractor (faza 1 z eksperymentu 05)."""
    X_train = torch.FloatTensor(data["X_train"]).unsqueeze(-1)
    y_train = torch.FloatTensor(data["y_train"])
    X_val = torch.FloatTensor(data["X_val"]).unsqueeze(-1)
    y_val = torch.FloatTensor(data["y_val"])

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)

    tcn = TCNFeatureExtractor()
    fc_proxy = nn.Linear(3, 1)

    params = list(tcn.parameters()) + list(fc_proxy.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience, patience_count = 15, 0

    print("  Trening TCN feature extractor...")
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
                print(f"    TCN early stopping epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS}, val_loss: {val_loss:.6f}")

    tcn.load_state_dict(best_tcn_state)
    tcn.eval()
    return tcn


# ============================================================================
# ALGORYTM GENETYCZNY
# ============================================================================

# Parametry GA
POP_SIZE = 30
NUM_GENERATIONS = 50
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.1
TOURNAMENT_K = 3
MUTATION_STD_MF = 0.15      # odchylenie standardowe mutacji parametrow MF
MUTATION_STD_RULE = 0.3      # odchylenie standardowe mutacji bitow regul
ELITISM = 2                  # liczba najlepszych osobnikow kopiowanych bez zmian


def evaluate_fitness(chrom, val_features, X_val_scaled, y_val_actual, scaler):
    """Oblicz fitness = -MAPE na zbiorze walidacyjnym."""
    mf_params, rule_active = decode_chromosome(chrom)

    sim = build_mamdani_from_params(mf_params, rule_active)
    if sim is None:
        return -100.0  # kara za zbyt malo regul

    predictions_scaled = []
    for i in range(len(val_features)):
        pred_change = mamdani_predict(sim, val_features[i])
        last_val = X_val_scaled[i, -1]
        pred_val = last_val * (1 + pred_change / 100)
        predictions_scaled.append(pred_val)

    predictions_scaled = np.array(predictions_scaled)
    predictions = inverse_transform(scaler, predictions_scaled)

    mape_val = compute_all_metrics(y_val_actual, predictions)["MAPE"]
    return -mape_val  # GA maksymalizuje fitness, wiec negujemy MAPE


def tournament_selection(population, fitness, k=TOURNAMENT_K):
    """Selekcja turniejowa."""
    selected = []
    for _ in range(len(population)):
        candidates = np.random.choice(len(population), size=k, replace=False)
        best = candidates[np.argmax(fitness[candidates])]
        selected.append(population[best].copy())
    return selected


def uniform_crossover(parent1, parent2):
    """Krzyzowanie jednolite."""
    child1 = parent1.copy()
    child2 = parent2.copy()
    mask = np.random.random(len(parent1)) < 0.5
    child1[mask] = parent2[mask]
    child2[mask] = parent1[mask]
    return child1, child2


def mutate(chrom):
    """Mutacja: szum gaussowski dla parametrow MF, perturbacja bitow regul."""
    mutated = chrom.copy()

    # Mutacja parametrow MF
    for i in range(NUM_MF_PARAMS):
        if np.random.random() < MUTATION_RATE:
            mutated[i] += np.random.normal(0, MUTATION_STD_MF)
            # Ogranicz szerokosci MF do wartosci dodatnich
            if i % 2 == 1:  # indeksy parzyste = centra, nieparzyste = szerokosci
                mutated[i] = max(mutated[i], 0.1)

    # Mutacja bitow regul
    for i in range(NUM_MF_PARAMS, CHROM_LEN):
        if np.random.random() < MUTATION_RATE:
            mutated[i] += np.random.normal(0, MUTATION_STD_RULE)
            mutated[i] = np.clip(mutated[i], 0.0, 1.0)

    return mutated


def run_ga(val_features, X_val_scaled, y_val_actual, scaler):
    """Glowna petla algorytmu genetycznego."""

    t_ga_start = time.time()

    # Inicjalizacja populacji
    population = []
    for i in range(POP_SIZE):
        if i == 0:
            # Pierwszy osobnik = domyslne parametry (reczne reguly)
            chrom = np.concatenate([DEFAULT_MF_PARAMS, DEFAULT_RULE_SELECTION])
        else:
            # Losowe perturbacje domyslnych parametrow
            mf_noise = DEFAULT_MF_PARAMS + np.random.normal(0, 0.3, NUM_MF_PARAMS)
            # Upewnij sie ze szerokosci sa dodatnie
            for j in range(1, NUM_MF_PARAMS, 2):
                mf_noise[j] = max(mf_noise[j], 0.1)
            rule_noise = DEFAULT_RULE_SELECTION + np.random.normal(0, 0.2, NUM_RULES)
            rule_noise = np.clip(rule_noise, 0.0, 1.0)
            chrom = np.concatenate([mf_noise, rule_noise])
        population.append(chrom)

    population = np.array(population)
    best_fitness_history = []
    mean_fitness_history = []

    print(f"\n  GA: populacja={POP_SIZE}, generacje={NUM_GENERATIONS}", flush=True)
    print(f"      krzyzowanie={CROSSOVER_RATE}, mutacja={MUTATION_RATE}", flush=True)

    best_ever_fitness = -float("inf")
    best_ever_chrom = None

    for gen in range(NUM_GENERATIONS):
        # Ewaluacja fitness z progressem
        fitness_list = []
        for ci, chrom in enumerate(population):
            f = evaluate_fitness(chrom, val_features, X_val_scaled, y_val_actual, scaler)
            fitness_list.append(f)
            print(f"\r    Gen {gen+1}/{NUM_GENERATIONS} - ewaluacja osobnika {ci+1}/{POP_SIZE}...", end="", flush=True)
        print("", flush=True)  # nowa linia
        fitness = np.array(fitness_list)

        # Zapisz najlepszego
        gen_best_idx = np.argmax(fitness)
        gen_best_fitness = fitness[gen_best_idx]

        if gen_best_fitness > best_ever_fitness:
            best_ever_fitness = gen_best_fitness
            best_ever_chrom = population[gen_best_idx].copy()

        best_fitness_history.append(gen_best_fitness)
        mean_fitness_history.append(np.mean(fitness))

        # Raport co kazda generacje z timestampem
        active = int(np.sum(decode_chromosome(population[gen_best_idx])[1]))
        now = datetime.datetime.now().strftime("%H:%M:%S")
        elapsed_gen = time.time() - t_ga_start
        print(f"    [{now}] Gen {gen+1:3d}/{NUM_GENERATIONS}: "
              f"MAPE={-gen_best_fitness:.4f}% (best_ever={-best_ever_fitness:.4f}%), "
              f"mean={-np.mean(fitness):.4f}%, "
              f"rules={active}/15, "
              f"elapsed={elapsed_gen:.0f}s", flush=True)

        # Elityzm
        elite_indices = np.argsort(fitness)[-ELITISM:]
        elites = [population[i].copy() for i in elite_indices]

        # Selekcja turniejowa
        selected = tournament_selection(population, fitness)

        # Krzyzowanie
        new_population = list(elites)  # zachowaj elity
        while len(new_population) < POP_SIZE:
            idx1, idx2 = np.random.choice(len(selected), size=2, replace=False)
            if np.random.random() < CROSSOVER_RATE:
                child1, child2 = uniform_crossover(selected[idx1], selected[idx2])
            else:
                child1, child2 = selected[idx1].copy(), selected[idx2].copy()

            # Mutacja
            child1 = mutate(child1)
            child2 = mutate(child2)

            new_population.append(child1)
            if len(new_population) < POP_SIZE:
                new_population.append(child2)

        population = np.array(new_population[:POP_SIZE])

    return best_ever_chrom, best_ever_fitness, best_fitness_history, mean_fitness_history


# ============================================================================
# GLOWNA PETLA
# ============================================================================

def main():
    dataset_key = "SP500"
    print(f"\n{'='*60}")
    print(f"  EKSPERYMENT 19: GA OPTYMALIZACJA REGUL MAMDANIEGO")
    print(f"  Zbior danych: {DATASETS[dataset_key]['name']}")
    print(f"{'='*60}")
    t0 = time.time()

    # 1. Przygotuj dane
    data = prepare_data(dataset_key)
    scaler = data["scaler"]

    # 2. Wytrenuj TCN feature extractor
    tcn = train_tcn_extractor(data)

    # 3. Wyciagnij cechy TCN na val i test
    X_val_t = torch.FloatTensor(data["X_val"]).unsqueeze(-1)
    X_test_t = torch.FloatTensor(data["X_test"]).unsqueeze(-1)

    with torch.no_grad():
        val_features = tcn(X_val_t).numpy()
        test_features = tcn(X_test_t).numpy()

    # Wartosci rzeczywiste na walidacji (do fitness) i tescie (do porownania)
    y_val_actual = inverse_transform(scaler, data["y_val"])
    y_test_actual = inverse_transform(scaler, data["y_test"])

    # 4. Ewaluacja recznie zdefiniowanych regul (baseline)
    print("\n  --- Baseline: reczne reguly (15/15 aktywnych) ---")
    baseline_chrom = np.concatenate([DEFAULT_MF_PARAMS, DEFAULT_RULE_SELECTION])
    baseline_fitness_val = evaluate_fitness(
        baseline_chrom, val_features, data["X_val"], y_val_actual, scaler
    )
    print(f"  Baseline MAPE (val): {-baseline_fitness_val:.4f}%")

    # Ewaluacja baseline na tescie
    mf_params_base, rule_active_base = decode_chromosome(baseline_chrom)
    sim_base = build_mamdani_from_params(mf_params_base, rule_active_base)
    preds_base_scaled = []
    for i in range(len(test_features)):
        pred_change = mamdani_predict(sim_base, test_features[i])
        last_val = data["X_test"][i, -1]
        preds_base_scaled.append(last_val * (1 + pred_change / 100))

    preds_base = inverse_transform(scaler, np.array(preds_base_scaled))
    baseline_metrics_test = compute_all_metrics(y_test_actual, preds_base)
    print(f"  Baseline MAPE (test): {baseline_metrics_test['MAPE']}%")

    # 5. Uruchom GA
    print("\n  --- Optymalizacja GA ---")
    best_chrom, best_fitness, fitness_hist, mean_hist = run_ga(
        val_features, data["X_val"], y_val_actual, scaler
    )

    # 6. Ewaluacja GA-optymalizowanych regul na tescie
    print("\n  --- Wyniki GA-optymalizowane ---")
    mf_params_ga, rule_active_ga = decode_chromosome(best_chrom)
    sim_ga = build_mamdani_from_params(mf_params_ga, rule_active_ga)

    preds_ga_scaled = []
    for i in range(len(test_features)):
        pred_change = mamdani_predict(sim_ga, test_features[i])
        last_val = data["X_test"][i, -1]
        preds_ga_scaled.append(last_val * (1 + pred_change / 100))

    preds_ga = inverse_transform(scaler, np.array(preds_ga_scaled))
    ga_metrics_test = compute_all_metrics(y_test_actual, preds_ga)

    elapsed = time.time() - t0

    # 7. Podsumowanie
    active_rules_list = [i + 1 for i, a in enumerate(rule_active_ga) if a]
    inactive_rules_list = [i + 1 for i, a in enumerate(rule_active_ga) if not a]

    rule_labels = [
        "R1:  trend=duzy_wzrost AND momentum=duzy_wzrost -> duzy_wzrost",
        "R2:  trend=wzrost AND momentum=wzrost -> wzrost",
        "R3:  trend=spadek AND momentum=spadek -> spadek",
        "R4:  trend=duzy_spadek AND momentum=duzy_spadek -> duzy_spadek",
        "R5:  trend=wzrost AND momentum=spadek -> neutralny",
        "R6:  trend=spadek AND momentum=wzrost -> neutralny",
        "R7:  trend=neutralny AND momentum=neutralny -> neutralny",
        "R8:  trend=neutralny AND momentum=wzrost -> wzrost",
        "R9:  trend=neutralny AND momentum=spadek -> spadek",
        "R10: trend=duzy_wzrost AND zmiennosc=wysoka -> neutralny",
        "R11: trend=duzy_spadek AND zmiennosc=wysoka -> neutralny",
        "R12: trend=wzrost AND zmiennosc=niska -> duzy_wzrost",
        "R13: trend=spadek AND zmiennosc=niska -> duzy_spadek",
        "R14: momentum=duzy_wzrost AND zmiennosc=srednia -> wzrost",
        "R15: momentum=duzy_spadek AND zmiennosc=srednia -> spadek",
    ]

    print(f"\n  Aktywne reguly GA ({len(active_rules_list)}/15):")
    for idx in active_rules_list:
        print(f"    {rule_labels[idx-1]}")
    if inactive_rules_list:
        print(f"  Wylaczone reguly: {inactive_rules_list}")

    print(f"\n  POROWNANIE (test set):")
    print(f"    Reczne reguly:       MAPE = {baseline_metrics_test['MAPE']}%")
    print(f"    GA-optymalizowane:   MAPE = {ga_metrics_test['MAPE']}%")
    improvement = baseline_metrics_test['MAPE'] - ga_metrics_test['MAPE']
    print(f"    Poprawa:             {improvement:.4f} pp")
    print(f"    Czas calkowity:      {elapsed:.1f}s")

    # 8. Zapis wynikow
    # Przygotuj zoptymalizowane parametry MF do zapisu
    mf_names = [
        "trend_duzy_spadek", "trend_spadek", "trend_neutralny",
        "trend_wzrost", "trend_duzy_wzrost",
        "zmiennosc_niska", "zmiennosc_srednia", "zmiennosc_wysoka",
        "momentum_duzy_spadek", "momentum_spadek", "momentum_neutralny",
        "momentum_wzrost", "momentum_duzy_wzrost",
        "prognoza_duzy_spadek", "prognoza_spadek", "prognoza_neutralny",
        "prognoza_wzrost", "prognoza_duzy_wzrost",
    ]
    optimized_mf = {}
    for i, name in enumerate(mf_names):
        optimized_mf[name] = {
            "center": round(float(mf_params_ga[i * 2]), 4),
            "width": round(float(mf_params_ga[i * 2 + 1]), 4),
        }

    result_data = {
        "model": "GA-optimized TCN+Mamdani",
        "dataset": dataset_key,
        "ga_parametry": {
            "populacja": POP_SIZE,
            "generacje": NUM_GENERATIONS,
            "krzyzowanie": CROSSOVER_RATE,
            "mutacja": MUTATION_RATE,
            "selekcja": f"turniejowa (k={TOURNAMENT_K})",
            "elityzm": ELITISM,
        },
        "baseline_reczne": {
            "MAPE_val": round(-baseline_fitness_val, 4),
            "MAPE_test": baseline_metrics_test["MAPE"],
            "RMSE_test": baseline_metrics_test["RMSE"],
            "MAE_test": baseline_metrics_test["MAE"],
            "aktywne_reguly": 15,
        },
        "ga_optymalizowane": {
            "MAPE_val": round(-best_fitness, 4),
            "MAPE_test": ga_metrics_test["MAPE"],
            "RMSE_test": ga_metrics_test["RMSE"],
            "MAE_test": ga_metrics_test["MAE"],
            "aktywne_reguly": len(active_rules_list),
            "reguly_aktywne": active_rules_list,
            "reguly_wylaczone": inactive_rules_list,
        },
        "poprawa_MAPE_pp": round(improvement, 4),
        "optimized_mf_params": optimized_mf,
        "fitness_history": {
            "best": [round(f, 4) for f in fitness_hist],
            "mean": [round(f, 4) for f in mean_hist],
        },
        "czas_s": round(elapsed, 2),
    }
    save_results(result_data, os.path.join(os.path.dirname(__file__), "wyniki.json"))
    print("\n=== GA OPTYMALIZACJA ZAKONCZONA ===")


if __name__ == "__main__":
    main()
