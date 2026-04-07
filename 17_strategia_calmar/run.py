"""
Eksperyment 17: Strategia inwestycyjna + Calmar ratio.
Symuluje prosta strategie: kup jesli model prognozuje wzrost, sprzedaj jesli spadek.
Porownuje zysk i Calmar ratio dla kazdego modelu.
Calmar = annualized return / max drawdown — metryka stosowana przez Wilinskiego.
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils import save_results
from config import DATASETS

OUT = os.path.dirname(os.path.abspath(__file__))
PRED_DIRS = {
    "ARIMA": os.path.join(os.path.dirname(OUT), "01_arima_baseline"),
    "LSTM": os.path.join(os.path.dirname(OUT), "02_lstm_baseline"),
    "TCN": os.path.join(os.path.dirname(OUT), "03_tcn_baseline"),
    "Mamdani": os.path.join(os.path.dirname(OUT), "04_mamdani_baseline"),
    "TCN-Mamdani": os.path.join(os.path.dirname(OUT), "05_tcn_mamdani_hybrid"),
}

COLORS = {"ARIMA": "#4472C4", "LSTM": "#ED7D31", "TCN": "#A5A5A5",
          "Mamdani": "#FFC000", "TCN-Mamdani": "#70AD47"}


def simulate_strategy(actual, predicted, initial_capital=10000):
    """
    Strategia: jesli predicted[t+1] > actual[t] -> kup (long)
               jesli predicted[t+1] < actual[t] -> sprzedaj (short)
    Zwraca: kapital skumulowany, Calmar ratio, max drawdown.
    """
    capital = initial_capital
    capital_history = [capital]
    n_trades = 0
    n_correct = 0

    for i in range(len(predicted) - 1):
        # Prognoza kierunku
        pred_direction = predicted[i] - actual[i]  # >0 = prognoza wzrostu
        actual_change = actual[i + 1] - actual[i]  # rzeczywista zmiana
        pct_change = actual_change / actual[i]

        if pred_direction > 0:
            # Long — zarabiamy jesli cena rosnie
            capital *= (1 + pct_change)
        else:
            # Short — zarabiamy jesli cena spada
            capital *= (1 - pct_change)

        capital_history.append(capital)
        n_trades += 1
        if (pred_direction > 0 and actual_change > 0) or (pred_direction <= 0 and actual_change <= 0):
            n_correct += 1

    capital_history = np.array(capital_history)

    # Max drawdown
    peak = np.maximum.accumulate(capital_history)
    drawdown = (peak - capital_history) / peak
    max_drawdown = np.max(drawdown)

    # Annualized return (zakladamy 252 dni handlowych/rok)
    total_return = (capital_history[-1] / initial_capital) - 1
    n_days = len(capital_history) - 1
    annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1

    # Calmar ratio
    calmar = annual_return / max_drawdown if max_drawdown > 0 else float("inf")

    # Buy & hold benchmark
    bh_return = (actual[-1] / actual[0]) - 1
    bh_capital = initial_capital * (1 + bh_return)

    return {
        "final_capital": round(capital, 2),
        "total_return_pct": round(total_return * 100, 2),
        "annual_return_pct": round(annual_return * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "calmar_ratio": round(calmar, 4),
        "n_trades": n_trades,
        "accuracy_pct": round(n_correct / max(n_trades, 1) * 100, 2),
        "buy_hold_capital": round(bh_capital, 2),
        "buy_hold_return_pct": round(bh_return * 100, 2),
        "capital_history": capital_history.tolist(),
    }


RESULTS = {}

for ds_key in DATASETS:
    ds_name = DATASETS[ds_key]["name"]
    print(f"\n=== {ds_name} ===")
    ds_results = {}

    fig, ax = plt.subplots(figsize=(10, 5))

    for model_name, model_dir in PRED_DIRS.items():
        pred_file = os.path.join(model_dir, f"prognozy_{ds_key}.csv")
        if not os.path.exists(pred_file):
            continue

        df = pd.read_csv(pred_file)
        actual = df["Actual"].values
        predicted = df["Predicted"].values

        result = simulate_strategy(actual, predicted)
        ds_results[model_name] = {k: v for k, v in result.items() if k != "capital_history"}

        print(f"  {model_name:15s}: kapital={result['final_capital']:>10.2f}  "
              f"return={result['total_return_pct']:>6.1f}%  "
              f"calmar={result['calmar_ratio']:>6.2f}  "
              f"maxDD={result['max_drawdown_pct']:>5.1f}%  "
              f"acc={result['accuracy_pct']:>5.1f}%")

        ax.plot(result["capital_history"], label=model_name,
                color=COLORS.get(model_name, "gray"), linewidth=1.2)

    # Buy & hold
    first_result = list(ds_results.values())[0]
    print(f"  {'Buy&Hold':15s}: kapital={first_result['buy_hold_capital']:>10.2f}  "
          f"return={first_result['buy_hold_return_pct']:>6.1f}%")

    ax.axhline(y=10000, color="black", linestyle="--", alpha=0.3, label="Kapital poczatkowy")
    ax.set_xlabel("Dzien handlowy")
    ax.set_ylabel("Kapital [USD/PLN]")
    ax.set_title(f"Skumulowany kapital strategii inwestycyjnej — {ds_name}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(OUT))),
                            "images", f"rys6_strategia_{ds_key}.png")
    plt.savefig(img_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Wykres: {img_path}")

    RESULTS[ds_key] = ds_results

# Podsumowanie
print("\n=== PODSUMOWANIE CALMAR RATIO ===")
print(f"{'Model':15s}", end="")
for ds in DATASETS:
    print(f"  {DATASETS[ds]['name']:>10s}", end="")
print()
print("-" * 60)
for model in PRED_DIRS:
    print(f"{model:15s}", end="")
    for ds in DATASETS:
        if model in RESULTS.get(ds, {}):
            print(f"  {RESULTS[ds][model]['calmar_ratio']:>10.2f}", end="")
        else:
            print(f"  {'---':>10s}", end="")
    print()

save_results({"eksperyment": "Strategia inwestycyjna + Calmar ratio", "wyniki": RESULTS},
             os.path.join(OUT, "wyniki.json"))
print("\n=== EKSPERYMENT 17 ZAKONCZONY ===")
