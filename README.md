# Pakiet reprodukowalności — TCN-Mamdani

Kod źródłowy eksperymentów do artykułu *„Integracja modeli sekwencyjnych i logiki rozmytej w analizie danych czasowych"* (hybryda TCN + system wnioskowania rozmytego Mamdaniego do prognozowania finansowych szeregów czasowych).

## Środowisko

- **Python 3.14**
- Instalacja zależności:

```bash
python -m pip install -r requirements.txt
```

Dokładne wersje bibliotek użytych do wygenerowania wyników są przypięte w [`requirements.txt`](requirements.txt). Wszystkie eksperymenty są deterministyczne (`RANDOM_SEED = 42`, ustawiany w `config.py` oraz `torch.manual_seed`/`np.random.seed` na początku każdego skryptu).

## Dane wejściowe

Dzienne ceny zamknięcia 7 instrumentów w `../zewnetrzne/<ZBIOR>/`:

| Zbiór | Plik | Źródło |
|---|---|---|
| S&P 500 | `SP500/sp500_daily.csv` | Yahoo Finance (yfinance) |
| WIG20 | `WIG20/wig20_d.csv` | Stooq.pl |
| EUR/USD | `EUR_USD/eurusd_daily.csv` | Yahoo Finance |
| BTC/USD | `BTC_USD/btcusd_daily.csv` | Yahoo Finance (od 2018) |
| DAX | `DAX/dax_daily.csv` | Yahoo Finance |
| Nikkei 225 | `NIKKEI/nikkei_daily.csv` | Yahoo Finance |
| Złoto | `GOLD/gold_daily.csv` | Yahoo Finance |

Podział train/val/test = 70/15/15 (chronologiczny, bez tasowania); okno wejściowe `LOOKBACK = 30`, horyzont `HORIZON = 1`. Szczegóły każdego eksperymentu w `NN_*/zrodlo.txt`.

## Struktura i kolejność uruchamiania

Każdy eksperyment to samodzielny `NN_nazwa/run.py`, zapisujący `wyniki.json`, `prognozy_<ZBIOR>.csv` oraz `zrodlo.txt`. Wspólny kod: `config.py` (parametry, ścieżki), `utils.py` (ładowanie danych, metryki MAPE/RMSE/MAE, test Diebolda-Mariano).

**Modele bazowe i hybryda (trenują od zera):**
`01_arima_baseline`, `02_lstm_baseline`, `03_tcn_baseline`, `04_mamdani_baseline`, `05_tcn_mamdani_hybrid`, `18_patchtst_benchmark`, `20_all_7datasets`.

**Analizy pogłębione (trenują DL):**
`06_ablacja`, `08_multi_step`, `09_stabilnosc`, `11_rozne_lookback`, `12_analiza_regul`, `13_zlozonosc_obliczeniowa`, `19_ga_reguly_optymalizacja`, `21_walk_forward`.

**Analizy na prognozach (czytają `prognozy_*.csv`, nie trenują) — uruchamiać PO powyższych:**
`14_test_diebold_mariano`, `15_directional_accuracy`, `16_kryzys_covid`, `17_strategia_calmar`.

Notebook `badania.ipynb` zawiera kod wszystkich eksperymentów 1:1 z `run.py` (samowystarczalny, do uruchomienia end-to-end).

## Uwagi metodologiczne (wersja 2026-06-28)

- **Wczesne zatrzymanie:** najlepsze wagi walidacyjne zapisywane przez `copy.deepcopy(model.state_dict())` (wcześniej płytka kopia `.copy()` aliasowała żywe tensory i przywracała wagi z ostatniej epoki — naprawione, modele przeliczone).
- **Test Diebolda-Mariano (`14`):** prognozy modeli wyrównywane po **dacie** (inner merge na kolumnie `Date` w `prognozy_*.csv`), nie po prefiksie tablic — porównywane są te same dni handlowe.
- **Strategia inwestycyjna (`17`):** sygnał bez look-ahead — cena znana w chwili decyzji to ostatni zrealizowany kurs `actual[t-1]`; jest to słaba diagnostyka pomocnicza (brak kosztów transakcyjnych, poślizgu, ograniczeń shortu), nie backtest inwestycyjny.

## Mapowanie do artykułu

Liczby w `artykul.md` (tabele, metryki inline) pochodzą **wyłącznie** z `NN_*/wyniki.json`. Generator obrazów wykresów: `../generate_figures.py` (paleta projektowa). Jedynym źródłem prawdy dla wartości liczbowych są pliki `wyniki.json`.
