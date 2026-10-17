# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 27 (czesc: epoki). Kontrola, w ktorej epoce zatrzymal sie trening ekstraktora TCN
w eksperymentach 05 i 20 (wczesne zatrzymanie, cierpliwosc 15).

Skrypty 05/run.py i 20/run.py sa uruchamiane BEZ ZMIAN z kopii w katalogu tymczasowym
(zeby nie nadpisac zapisanych wynikow), a ich wyjscie trafia do log_powtorka_05.txt
i log_powtorka_20.txt. Skrypt odczytuje z logu komunikat "Early stopping epoch N";
najlepsza epoka walidacyjna = N - 15. Na koniec porownuje MAPE z powtorki z MAPE zapisanym
w 05/wyniki.json i 20/wyniki.json (zgodnosc = log dotyczy tego samego przebiegu).
"""
import os, re, sys, json, shutil, subprocess, tempfile

OUT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(OUT)
sys.path.append(BASE)
from config import DATASETS
from utils import save_results
PATIENCE = 15
SRC = {"05": ("05_tcn_mamdani_hybrid", "wyniki"), "20": ("20_all_7datasets", "wyniki_hybrid")}


def parse(log):
    out, cur = {}, None
    for line in log.splitlines():
        m = re.match(r"^=== (TCN-Mamdani: |ARIMA: )?(.+?) ===$", line.strip())
        if m:
            cur = None if m.group(1) == "ARIMA: " else m.group(2)
            if cur:
                out.setdefault(cur, {"stop": None, "mape": None})
            continue
        m = re.search(r"Early stopping epoch (\d+)", line)
        if m and cur:
            out[cur]["stop"] = int(m.group(1))
        m = re.search(r"MAPE: ([0-9.]+)%", line)
        if m and cur and out[cur]["mape"] is None:
            out[cur]["mape"] = float(m.group(1))
    return out


wynik = {}
for exp, (folder, key) in SRC.items():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, folder)
        os.makedirs(d)
        shutil.copy(os.path.join(BASE, folder, "run.py"), d)
        env = dict(os.environ, PYTHONPATH=BASE)
        p = subprocess.run([sys.executable, os.path.join(d, "run.py")], cwd=d, env=env,
                           capture_output=True, text=True)
    log = p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr.strip() else "")
    with open(os.path.join(OUT, f"log_powtorka_{exp}.txt"), "w", encoding="utf-8") as f:
        f.write(log)
    stored = json.load(open(os.path.join(BASE, folder, "wyniki.json"), encoding="utf-8"))[key]
    names = {v["name"]: k for k, v in DATASETS.items()}
    res = {}
    for name, v in parse(log).items():
        k = names.get(name)
        if k is None or k not in stored:
            continue
        stop = v["stop"]
        res[k] = {
            "epoka_zatrzymania": stop,
            "najlepsza_epoka_walidacyjna": (stop - PATIENCE) if stop else "brak wczesnego zatrzymania (100 epok)",
            "MAPE_powtorka": v["mape"],
            "MAPE_zapisane": stored[k]["MAPE"],
            "zgodne": v["mape"] is not None and abs(v["mape"] - stored[k]["MAPE"]) < 1e-4,
        }
        print(exp, k, res[k], flush=True)
    wynik[exp] = res

save_results({"eksperyment": "27: epoka wczesnego zatrzymania TCN w 05 i 20 (powtorka bez zmian kodu)",
              "cierpliwosc": PATIENCE, "wyniki": wynik}, os.path.join(OUT, "wyniki_epoki.json"))
