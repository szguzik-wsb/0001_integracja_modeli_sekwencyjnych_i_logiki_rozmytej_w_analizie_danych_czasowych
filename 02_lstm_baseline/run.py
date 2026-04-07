"""Eksperyment 02: LSTM baseline na 4 zbiorach danych."""
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


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze()


def train_and_evaluate(dataset_key):
    print(f"\n=== {DATASETS[dataset_key]['name']} ===")
    t0 = time.time()

    data = prepare_data(dataset_key)
    X_train = torch.FloatTensor(data["X_train"]).unsqueeze(-1)
    y_train = torch.FloatTensor(data["y_train"])
    X_val = torch.FloatTensor(data["X_val"]).unsqueeze(-1)
    y_val = torch.FloatTensor(data["y_val"])
    X_test = torch.FloatTensor(data["X_test"]).unsqueeze(-1)
    y_test = torch.FloatTensor(data["y_test"])

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)

    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    patience, patience_count = 15, 0

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

    predictions = inverse_transform(data["scaler"], predictions_scaled)
    actuals = inverse_transform(data["scaler"], data["y_test"])
    elapsed = time.time() - t0

    metrics = compute_all_metrics(actuals, predictions)
    metrics["czas_s"] = round(elapsed, 2)
    metrics["best_epoch"] = EPOCHS - patience_count

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

save_results({"model": "LSTM(hidden=64, layers=2)", "wyniki": RESULTS},
             os.path.join(os.path.dirname(__file__), "wyniki.json"))
print("\n=== LSTM BASELINE ZAKONCZONE ===")
