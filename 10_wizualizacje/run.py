# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 10: Wizualizacje wynikow eksperymentow 01-05.

Generuje trzy wykresy dla artykulu:
  1. Actual vs Predicted — porownanie 5 modeli na S&P 500 (ostatnie 100 dni)
  2. Rozklad bledow predykcji (histogramy) dla kazdego modelu
  3. Funkcje przynaleznosci systemu Mamdaniego (trend, zmiennosc, momentum, prognoza)

Wykresy zapisywane sa jako PNG w podfolderze wykresy/.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skfuzzy as fuzz

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(EXP_DIR)
WYKRESY_DIR = os.path.join(EXP_DIR, "wykresy")
os.makedirs(WYKRESY_DIR, exist_ok=True)

# Konfiguracja stylu wykresow
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# === LADOWANIE PROGNOZ Z EKSPERYMENTOW 01-05 ===

MODELS = {
    "ARIMA": os.path.join(PARENT_DIR, "01_arima_baseline", "prognozy_SP500.csv"),
    "LSTM": os.path.join(PARENT_DIR, "02_lstm_baseline", "prognozy_SP500.csv"),
    "TCN": os.path.join(PARENT_DIR, "03_tcn_baseline", "prognozy_SP500.csv"),
    "Mamdani": os.path.join(PARENT_DIR, "04_mamdani_baseline", "prognozy_SP500.csv"),
    "TCN+Mamdani": os.path.join(PARENT_DIR, "05_tcn_mamdani_hybrid", "prognozy_SP500.csv"),
}

COLORS = {
    "ARIMA": "#1f77b4",
    "LSTM": "#ff7f0e",
    "TCN": "#2ca02c",
    "Mamdani": "#d62728",
    "TCN+Mamdani": "#9467bd",
}

print("Ladowanie prognoz...")
data = {}
for name, path in MODELS.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        data[name] = df
        print(f"  {name}: {len(df)} wierszy")
    else:
        print(f"  {name}: BRAK PLIKU ({path})")

if not data:
    print("BLAD: Brak danych do wizualizacji. Uruchom najpierw eksperymenty 01-05.")
    sys.exit(1)


# === WYKRES 1: ACTUAL VS PREDICTED (OSTATNIE 100 DNI) ===

print("\nGenerowanie wykresu 1: Actual vs Predicted...")

LAST_N = 100
fig, ax = plt.subplots(figsize=(14, 6))

# Wartosci rzeczywiste (z pierwszego dostepnego modelu)
min_len = min(len(df) for df in data.values())
first_model = list(data.keys())[0]
actuals = data[first_model]["Actual"].values[-min_len:]
n = min(min_len, LAST_N)
ax.plot(range(n), actuals[-n:],
        color="black", linewidth=2, label="Actual", zorder=10)

# Prognozy kazdego modelu
for name, df in data.items():
    preds = df["Predicted"].values[-min_len:]
    ax.plot(range(n), preds[-n:],
            color=COLORS.get(name, "gray"), linewidth=1.2,
            alpha=0.8, label=name)

ax.set_xlabel("Dzien (ostatnie 100 dni zbioru testowego)")
ax.set_ylabel("Cena zamkniecia S&P 500 [USD]")
ax.set_title("Porownanie prognoz 5 modeli na S&P 500 (ostatnie 100 dni)")
ax.legend(loc="best", framealpha=0.9)
ax.set_xlim(0, n - 1)

plt.tight_layout()
path1 = os.path.join(WYKRESY_DIR, "01_actual_vs_predicted.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Zapisano: {path1}")


# === WYKRES 2: ROZKLAD BLEDOW PREDYKCJI (HISTOGRAMY) ===

print("Generowanie wykresu 2: Rozklad bledow predykcji...")

n_models = len(data)
fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4), sharey=True)
if n_models == 1:
    axes = [axes]

for idx, (name, df) in enumerate(data.items()):
    errors = df["Actual"].values - df["Predicted"].values
    ax = axes[idx]
    ax.hist(errors, bins=50, color=COLORS.get(name, "gray"),
            alpha=0.75, edgecolor="black", linewidth=0.5)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1, alpha=0.7)

    mean_err = np.mean(errors)
    std_err = np.std(errors)
    ax.set_title(f"{name}\n(mean={mean_err:.2f}, std={std_err:.2f})")
    ax.set_xlabel("Blad predykcji [USD]")
    if idx == 0:
        ax.set_ylabel("Liczba obserwacji")

fig.suptitle("Rozklad bledow predykcji na S&P 500", fontsize=14, y=1.02)
plt.tight_layout()
path2 = os.path.join(WYKRESY_DIR, "02_rozklad_bledow.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Zapisano: {path2}")


# === WYKRES 3: FUNKCJE PRZYNALEZNOSCI MAMDANIEGO ===

print("Generowanie wykresu 3: Funkcje przynaleznosci Mamdaniego...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# --- Trend ---
ax = axes[0, 0]
universe = np.linspace(-3, 3, 1000)
labels_5 = {
    "duzy_spadek": (-2.5, 0.6),
    "spadek": (-1.0, 0.4),
    "neutralny": (0.0, 0.35),
    "wzrost": (1.0, 0.4),
    "duzy_wzrost": (2.5, 0.6),
}
colors_mf = ["#d62728", "#ff7f0e", "#7f7f7f", "#2ca02c", "#1f77b4"]

for (label, (mean, sigma)), color in zip(labels_5.items(), colors_mf):
    mf = fuzz.gaussmf(universe, mean, sigma)
    ax.plot(universe, mf, label=label, color=color, linewidth=1.5)
    ax.fill_between(universe, mf, alpha=0.1, color=color)
ax.set_title("Trend")
ax.set_xlabel("Wartosc cechy")
ax.set_ylabel("Stopien przynaleznosci")
ax.legend(loc="upper left", fontsize=8)
ax.set_ylim(0, 1.05)

# --- Zmiennosc ---
ax = axes[0, 1]
universe_vol = np.linspace(0, 5, 1000)
labels_vol = {
    "niska": (0, 0.5),
    "srednia": (2, 0.6),
    "wysoka": (4, 0.7),
}
colors_vol = ["#2ca02c", "#ff7f0e", "#d62728"]

for (label, (mean, sigma)), color in zip(labels_vol.items(), colors_vol):
    mf = fuzz.gaussmf(universe_vol, mean, sigma)
    ax.plot(universe_vol, mf, label=label, color=color, linewidth=1.5)
    ax.fill_between(universe_vol, mf, alpha=0.1, color=color)
ax.set_title("Zmiennosc (Volatility)")
ax.set_xlabel("Wartosc cechy")
ax.set_ylabel("Stopien przynaleznosci")
ax.legend(loc="upper left", fontsize=8)
ax.set_ylim(0, 1.05)

# --- Momentum ---
ax = axes[1, 0]
for (label, (mean, sigma)), color in zip(labels_5.items(), colors_mf):
    mf = fuzz.gaussmf(universe, mean, sigma)
    ax.plot(universe, mf, label=label, color=color, linewidth=1.5)
    ax.fill_between(universe, mf, alpha=0.1, color=color)
ax.set_title("Momentum")
ax.set_xlabel("Wartosc cechy")
ax.set_ylabel("Stopien przynaleznosci")
ax.legend(loc="upper left", fontsize=8)
ax.set_ylim(0, 1.05)

# --- Prognoza (wyjscie) ---
ax = axes[1, 1]
universe_prog = np.linspace(-5, 5, 1000)
labels_prog = {
    "duzy_spadek": (-2.5, 0.6),
    "spadek": (-1.0, 0.4),
    "neutralny": (0.0, 0.35),
    "wzrost": (1.0, 0.4),
    "duzy_wzrost": (2.5, 0.6),
}

for (label, (mean, sigma)), color in zip(labels_prog.items(), colors_mf):
    mf = fuzz.gaussmf(universe_prog, mean, sigma)
    ax.plot(universe_prog, mf, label=label, color=color, linewidth=1.5)
    ax.fill_between(universe_prog, mf, alpha=0.1, color=color)
ax.set_title("Prognoza (wyjscie systemu Mamdaniego)")
ax.set_xlabel("Wartosc wyjsciowa")
ax.set_ylabel("Stopien przynaleznosci")
ax.legend(loc="upper left", fontsize=8)
ax.set_ylim(0, 1.05)

fig.suptitle("Funkcje przynaleznosci systemu Mamdaniego", fontsize=14)
plt.tight_layout()
path3 = os.path.join(WYKRESY_DIR, "03_funkcje_przynaleznosci.png")
plt.savefig(path3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Zapisano: {path3}")


print(f"\n=== EKSPERYMENT 10 ZAKONCZONY ===")
print(f"Wykresy zapisane w: {WYKRESY_DIR}")
