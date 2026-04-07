# Autor: mgr inz. Szymon Guzik, Uniwersytet WSB Merito w Gdansku
"""
Eksperyment 18: PatchTST — Transformer baseline na wszystkich 7 zbiorach danych.

PatchTST (Patch Time Series Transformer) dzieli szereg czasowy na patche
(fragmenty), projektuje je do przestrzeni embeddingowej i przetwarza
standardowym encoderem Transformer. Podejscie to pozwala na efektywne
modelowanie dlugoterminowych zaleznosci przy nizszym koszcie obliczeniowym
niz klasyczny Transformer operujacy na pojedynczych krokach czasowych.

Architektura:
  [Dane cenowe] -> [Patchowanie: len=8, stride=4]
                       |
                       v
                 [Linear projection -> d_model=32]
                       |
                       v
                 [Learnable positional encoding]
                       |
                       v
                 [TransformerEncoder: nhead=4, layers=2, ff=64]
                       |
                       v
                 [Linear head -> prognoza]

Zrodlo: Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023).
        A Time Series is Worth 64 Words. ICLR 2023.
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import prepare_data, compute_all_metrics, inverse_transform, save_results
from config import DATASETS, LOOKBACK, EPOCHS, BATCH_SIZE, LEARNING_RATE, RANDOM_SEED

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# === PATCHTST MODEL ===

class PatchTST(nn.Module):
    """Uproszczona implementacja PatchTST.

    Wejscie: (batch, seq_len, 1) — znormalizowany szereg cenowy.
    Patchowanie: unfold z patch_len i stride -> sekwencja patchow.
    Projekcja: Linear(patch_len, d_model).
    Pozycja: learnable positional encoding.
    Encoder: nn.TransformerEncoder.
    Wyjscie: Linear z ostatniego tokena na prognoze.
    """

    def __init__(self, seq_len=LOOKBACK, patch_len=8, stride=4,
                 d_model=32, nhead=4, num_layers=2, dim_feedforward=64,
                 dropout=0.2):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride

        # Oblicz liczbe patchow
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Projekcja patcha do d_model
        self.patch_projection = nn.Linear(patch_len, d_model)

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # LayerNorm przed head
        self.norm = nn.LayerNorm(d_model)

        # Linear head: z ostatniego patcha do prognozy
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, 1)
        """
        # (batch, seq_len)
        x = x.squeeze(-1)

        # Patchowanie za pomoca unfold: (batch, num_patches, patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # Projekcja liniowa: (batch, num_patches, d_model)
        patch_emb = self.patch_projection(patches)

        # Dodaj positional encoding
        patch_emb = patch_emb + self.pos_embedding

        # Transformer encoder
        encoded = self.transformer_encoder(patch_emb)

        # Normalizacja
        encoded = self.norm(encoded)

        # Wez ostatni patch token
        last_token = encoded[:, -1, :]

        # Prognoza
        out = self.head(last_token).squeeze(-1)
        return out


# === TRENING I EWALUACJA ===

def train_and_evaluate(dataset_key):
    print(f"\n=== {DATASETS[dataset_key]['name']} ===")
    t0 = time.time()

    data = prepare_data(dataset_key)
    X_train = torch.FloatTensor(data["X_train"]).unsqueeze(-1)
    y_train = torch.FloatTensor(data["y_train"])
    X_val = torch.FloatTensor(data["X_val"]).unsqueeze(-1)
    y_val = torch.FloatTensor(data["y_val"])
    X_test = torch.FloatTensor(data["X_test"]).unsqueeze(-1)

    scaler = data["scaler"]
    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)

    model = PatchTST(
        seq_len=LOOKBACK,
        patch_len=8,
        stride=4,
        d_model=32,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        dropout=0.2,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience, patience_count = 15, 0
    best_state = None

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
                print(f"  Early stopping epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS}, val_loss: {val_loss:.6f}")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        predictions_scaled = model(X_test).numpy()

    predictions = inverse_transform(scaler, predictions_scaled)
    actuals = inverse_transform(scaler, data["y_test"])
    elapsed = time.time() - t0

    metrics = compute_all_metrics(actuals, predictions)
    metrics["czas_s"] = round(elapsed, 2)
    metrics["best_epoch"] = EPOCHS - patience_count
    metrics["num_patches"] = model.num_patches

    print(f"  MAPE: {metrics['MAPE']}%")
    print(f"  RMSE: {metrics['RMSE']}")
    print(f"  MAE:  {metrics['MAE']}")
    print(f"  Czas: {metrics['czas_s']}s")
    print(f"  Patche: {model.num_patches}")

    return metrics


# === GLOWNA PETLA ===

RESULTS = {}
for ds_key in DATASETS:
    RESULTS[ds_key] = train_and_evaluate(ds_key)

result_data = {
    "model": "PatchTST(patch=8, stride=4, d=32, heads=4, layers=2, ff=64)",
    "architektura": "Patch embedding + TransformerEncoder + Linear head",
    "lookback": LOOKBACK,
    "wyniki": RESULTS,
}
save_results(result_data, os.path.join(os.path.dirname(__file__), "wyniki.json"))
print("\n=== PATCHTST BENCHMARK ZAKONCZONE ===")
