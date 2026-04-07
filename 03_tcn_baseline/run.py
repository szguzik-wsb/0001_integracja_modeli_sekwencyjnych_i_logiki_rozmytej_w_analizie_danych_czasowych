"""Eksperyment 03: TCN baseline na 4 zbiorach danych."""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils import prepare_data, compute_all_metrics, inverse_transform, save_results
from config import DATASETS, LOOKBACK, EPOCHS, BATCH_SIZE, LEARNING_RATE, RANDOM_SEED

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class CausalConv1d(nn.Module):
    """Konwolucja przyczynowa (causal) — nie widzi przyszlosci."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """Blok rezydualny TCN (Bai et al., 2018)."""
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


class TCNModel(nn.Module):
    """Temporal Convolutional Network do prognozowania szeregów czasowych."""
    def __init__(self, input_size=1, num_channels=[32, 32, 32], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        # x: (batch, seq_len, 1) -> (batch, 1, seq_len)
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = out[:, :, -1]  # ostatni krok czasowy
        return self.fc(out).squeeze()


def train_and_evaluate(dataset_key):
    print(f"\n=== {DATASETS[dataset_key]['name']} ===")
    t0 = time.time()

    data = prepare_data(dataset_key)
    X_train = torch.FloatTensor(data["X_train"]).unsqueeze(-1)
    y_train = torch.FloatTensor(data["y_train"])
    X_val = torch.FloatTensor(data["X_val"]).unsqueeze(-1)
    y_val = torch.FloatTensor(data["y_val"])
    X_test = torch.FloatTensor(data["X_test"]).unsqueeze(-1)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)

    model = TCNModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience, patience_count = 15, 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

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

    predictions = inverse_transform(data["scaler"], predictions_scaled)
    actuals = inverse_transform(data["scaler"], data["y_test"])
    elapsed = time.time() - t0

    metrics = compute_all_metrics(actuals, predictions)
    metrics["czas_s"] = round(elapsed, 2)

    print(f"  MAPE: {metrics['MAPE']}%")
    print(f"  RMSE: {metrics['RMSE']}")
    print(f"  MAE:  {metrics['MAE']}")
    print(f"  Czas: {metrics['czas_s']}s")

    pred_df = pd.DataFrame({"Actual": actuals, "Predicted": predictions})
    pred_df.to_csv(os.path.join(os.path.dirname(__file__), f"prognozy_{dataset_key}.csv"), index=False)

    return metrics


RESULTS = {}
for ds_key in DATASETS:
    RESULTS[ds_key] = train_and_evaluate(ds_key)

save_results({"model": "TCN(channels=[32,32,32], kernel=3)", "wyniki": RESULTS},
             os.path.join(os.path.dirname(__file__), "wyniki.json"))
print("\n=== TCN BASELINE ZAKONCZONE ===")
