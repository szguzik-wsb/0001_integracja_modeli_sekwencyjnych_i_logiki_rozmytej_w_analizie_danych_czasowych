"""Generowanie tabel jako PNG — v4: pelna siatka, klasyczny styl naukowy."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "images")
print(f"IMG_DIR: {IMG_DIR}")

PAGE_WIDTH = 7.5


def make_table(data, col_labels, title, filename, fontsize=10, col_widths=None, highlight_min_cols=None):
    n_rows = len(data)
    n_cols = len(col_labels)
    fig_height = max((n_rows + 1) * 0.4 + 0.9, 2.5)

    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, fig_height))
    ax.axis('off')

    table = ax.table(
        cellText=data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
        edges='closed',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.5)

    # Szerokosci kolumn
    if col_widths:
        for i in range(n_rows + 1):
            for j in range(n_cols):
                table[i, j].set_width(col_widths[j])
    else:
        w = 1.0 / n_cols
        for i in range(n_rows + 1):
            for j in range(n_cols):
                table[i, j].set_width(w)

    # Styl naglowkow
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor('#2F5496')
        cell.set_text_props(color='white', fontweight='bold', fontsize=fontsize)
        cell.set_edgecolor('#1a1a1a')
        cell.set_linewidth(0.8)

    # Styl wierszy
    for i in range(n_rows):
        for j in range(n_cols):
            cell = table[i + 1, j]
            cell.set_edgecolor('#666666')
            cell.set_linewidth(0.5)
            if j == 0:
                cell.set_facecolor('#D6E4F0')
                cell.set_text_props(fontweight='bold', fontsize=fontsize - 0.5)
            else:
                cell.set_facecolor('#F7F9FC' if i % 2 == 0 else '#FFFFFF')

    # Podswietlenie najlepszych
    if highlight_min_cols:
        for j in highlight_min_cols:
            vals = []
            for i in range(n_rows):
                try:
                    v = float(str(data[i][j]).replace(',', '.').replace('\u2014', '999').strip())
                    vals.append(v)
                except:
                    vals.append(999)
            best_i = int(np.argmin(vals))
            if vals[best_i] < 999:
                cell = table[best_i + 1, j]
                cell.set_text_props(fontweight='bold', color='#C00000')

    ax.set_title(title, fontsize=fontsize + 1, fontweight='bold', pad=12, loc='center')

    # Zrodlo pod tabela
    GITHUB_URL = "https://github.com/szguzik/tcn-mamdani-hybrid"
    ax.text(0.5, 0.02, f'Zrodlo: opracowanie wlasne. Kod: {GITHUB_URL}',
            transform=ax.transAxes, fontsize=7.5, ha='center', va='bottom',
            style='italic', color='#555555')

    plt.subplots_adjust(left=0.0, right=1.0, top=0.88, bottom=0.06)
    plt.savefig(os.path.join(IMG_DIR, filename), dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.05)
    plt.close()
    print(f"  {filename} saved")


# TABELA 1: Baza regul
make_table(
    data=[
        ['R1',  'trend = duzy_wzrost AND momentum = duzy_wzrost', 'duzy_wzrost'],
        ['R2',  'trend = wzrost AND momentum = wzrost', 'wzrost'],
        ['R3',  'trend = spadek AND momentum = spadek', 'spadek'],
        ['R4',  'trend = duzy_spadek AND momentum = duzy_spadek', 'duzy_spadek'],
        ['R5',  'trend = wzrost AND momentum = spadek', 'neutralny'],
        ['R6',  'trend = spadek AND momentum = wzrost', 'neutralny'],
        ['R7',  'trend = neutralny AND momentum = neutralny', 'neutralny'],
        ['R8',  'trend = neutralny AND momentum = wzrost', 'wzrost'],
        ['R9',  'trend = neutralny AND momentum = spadek', 'spadek'],
        ['R10', 'trend = duzy_wzrost AND zmiennosc = wysoka', 'neutralny'],
        ['R11', 'trend = duzy_spadek AND zmiennosc = wysoka', 'neutralny'],
        ['R12', 'trend = wzrost AND zmiennosc = niska', 'duzy_wzrost'],
        ['R13', 'trend = spadek AND zmiennosc = niska', 'duzy_spadek'],
        ['R14', 'momentum = duzy_wzrost AND zmiennosc = srednia', 'wzrost'],
        ['R15', 'momentum = duzy_spadek AND zmiennosc = srednia', 'spadek'],
    ],
    col_labels=['Nr', 'Przeslanka (IF)', 'Konkluzja (THEN)'],
    title='Tabela 1. Baza regul systemu Mamdaniego',
    filename='tab1_reguly.png',
    col_widths=[0.06, 0.7, 0.24],
)

# TABELA 2: Dane
make_table(
    data=[
        ['S&P 500', '2015\u20132025', '2 765', '1 829', '6 932', 'Yahoo Finance'],
        ['WIG20', '2015\u20132025', '2 749', '1 389', '3 484', 'Stooq.pl'],
        ['EUR/USD', '2015\u20132025', '2 862', '0,960', '1,251', 'Yahoo Finance'],
        ['BTC/USD', '2018\u20132025', '2 921', '3 237', '124 753', 'Yahoo Finance'],
    ],
    col_labels=['Zbior', 'Okres', 'Obserwacje', 'Close min', 'Close max', 'Zrodlo'],
    title='Tabela 2. Charakterystyka zbiorow danych',
    filename='tab2_dane.png',
    col_widths=[0.14, 0.16, 0.14, 0.14, 0.14, 0.18],
)

# TABELA 3: MAPE
make_table(
    data=[
        ['ARIMA', '0,682', '1,017', '0,332', '1,642', '0,746', '1,064', '0,903'],
        ['LSTM', '6,016', '1,316', '0,381', '9,468', '\u2014', '\u2014', '\u2014'],
        ['TCN', '4,453', '1,633', '0,374', '16,388', '\u2014', '\u2014', '\u2014'],
        ['PatchTST', '10,348', '3,163', '0,612', '26,248', '16,828', '17,941', '23,998'],
        ['Mamdani', '0,921', '1,255', '0,504', '1,731', '\u2014', '\u2014', '\u2014'],
        ['TCN-Mamdani', '1,382', '1,036', '0,339', '1,599', '1,391', '1,084', '1,374'],
    ],
    col_labels=['Model', 'S&P 500', 'WIG20', 'EUR/USD', 'BTC/USD', 'DAX', 'Nikkei', 'Gold'],
    title='Tabela 3. Porownanie MAPE [%] modeli na zbiorach testowych',
    filename='tab3_mape.png',
    highlight_min_cols=[1, 2, 3, 4, 5, 6, 7],
    col_widths=[0.16, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
)

# TABELA 4: DM test
make_table(
    data=[
        ['S&P 500', '\u22123,32 (0,001)*', '12,13 (<0,001)*', '14,21 (<0,001)*', '\u22122,08 (0,038)*'],
        ['WIG20', '\u22120,31 (0,757)', '2,67 (0,008)*', '4,65 (<0,001)*', '1,15 (0,250)'],
        ['EUR/USD', '\u22120,89 (0,372)', '\u22120,65 (0,517)', '\u22121,33 (0,185)', '2,27 (0,024)*'],
        ['BTC/USD', '\u22120,59 (0,557)', '16,38 (<0,001)*', '23,95 (<0,001)*', '\u22120,19 (0,850)'],
    ],
    col_labels=['Zbior', 'vs ARIMA', 'vs LSTM', 'vs TCN', 'vs Mamdani'],
    title='Tabela 4. Test Diebold-Mariano: TCN-Mamdani vs modele bazowe (DM, p)',
    filename='tab4_dm_test.png',
    col_widths=[0.14, 0.215, 0.215, 0.215, 0.215],
    fontsize=9.5,
)

# TABELA 5: Zlozonosc
make_table(
    data=[
        ['ARIMA', '\u2014', '0,12/obs.', '6,33'],
        ['LSTM', '50 497', '38,52', '0,005'],
        ['TCN', '16 129', '92,75', '0,006'],
        ['Mamdani', '0 (reguly)', '0,16', '2,08'],
        ['TCN-Mamdani', '9 859', '10,05', '4,41'],
    ],
    col_labels=['Model', 'Parametry', 'Trening [s]', 'Inferencja [s]'],
    title='Tabela 5. Zlozonosc obliczeniowa modeli (S&P 500)',
    filename='tab5_zlozonosc.png',
    col_widths=[0.25, 0.25, 0.25, 0.25],
)

# TABELA 6: Interpretowalnosc
make_table(
    data=[
        ['Typ', 'inherentna (ante-hoc)', 'post-hoc'],
        ['Czytelnosc', 'wysoka \u2014 reguly IF-THEN', 'niska \u2014 wartosci numeryczne'],
        ['Weryfikacja ekspercka', 'tak', 'ograniczona'],
        ['Zasieg', 'globalny', 'lokalny (per prognoza)'],
        ['Koszt obliczeniowy', 'zerowy', 'wysoki \u2014 O(d\u00b72\u207f)'],
        ['Zgodnosc z EU AI Act', 'pelna', 'czesciowa'],
    ],
    col_labels=['Kryterium', 'Mamdani (TCN-Mamdani)', 'SHAP (czysty TCN)'],
    title='Tabela 6. Porownanie interpretowalnosci: Mamdani vs SHAP',
    filename='tab6_interpretowalnosc.png',
    col_widths=[0.25, 0.375, 0.375],
)

print("\nWSZYSTKIE TABELE GOTOWE (v4 - pelna siatka)")
