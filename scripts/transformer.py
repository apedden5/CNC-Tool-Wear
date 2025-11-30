import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# ------------------------
# Config (fixed)
# ------------------------
WINDOW_SIZE = 10        # must match preprocessing
BATCH_SIZE = 64
MAX_EPOCHS = 15
EARLY_STOP_PATIENCE = 3
MAX_GRAD_NORM = 1.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if (PROJECT_ROOT / "data" / "data_windowed_csv").exists():
    DATA_ROOT = PROJECT_ROOT / "data"
else:
    DATA_ROOT = PROJECT_ROOT
WINDOWED_DIR = DATA_ROOT / "data_windowed_csv"

# ------------------------
# Dataset
# ------------------------
class ToolWearWindowDataset(Dataset):
    def __init__(self, csv_path: Path, window_size: int):
        self.df = pd.read_csv(csv_path)
        self.window_size = window_size

        self.labels = self.df["tool_condition"].values.astype(np.float32)

        drop_cols = ["tool_condition", "experiment_id"]
        for c in drop_cols:
            if c in self.df.columns:
                self.df = self.df.drop(columns=c)

        X = self.df.values.astype(np.float32)
        total_features = X.shape[1]
        assert total_features % window_size == 0, (
            f"Feature count {total_features} not divisible by window size {window_size}"
        )

        self.n_features = total_features // window_size
        self.X = X.reshape(-1, window_size, self.n_features)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.labels[idx]
        return torch.from_numpy(x), torch.tensor(y)


# ------------------------
# Model
# ------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ToolWearTransformer(nn.Module):
    def __init__(self, in_features, d_model, nhead, num_layers, dropout):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(d_model)

        self.cls_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.cls_head(x).squeeze(-1)


# ------------------------
# Training helpers
# ------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        running += loss.item() * x.size(0)

    return running / len(loader.dataset)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            running += loss.item() * x.size(0)

    return running / len(loader.dataset)


def eval_auc(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x).cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
            all_y.append(y.numpy())
            all_p.append(probs)
    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_p)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return auc, y_true, y_prob


# ------------------------
# HPO: try many configs
# ------------------------
def run_single_config(config, train_loader, val_loader, in_features, device):
    d_model      = config["d_model"]
    nhead        = config["nhead"]
    num_layers   = config["num_layers"]
    dropout      = config["dropout"]
    lr           = config["lr"]
    weight_decay = config["weight_decay"]

    model = ToolWearTransformer(
        in_features=in_features,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_val_loss = float("inf")
    best_state = None
    patience = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = eval_one_epoch(model, val_loader, criterion, device)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                break

    # restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    val_auc, _, _ = eval_auc(model, val_loader, device)
    return best_val_loss, val_auc, model


def main():
    # ---- data ----
    train_csv = WINDOWED_DIR / f"train_windows_w{WINDOW_SIZE}.csv"
    val_csv   = WINDOWED_DIR / f"val_windows_w{WINDOW_SIZE}.csv"

    train_ds = ToolWearWindowDataset(train_csv, WINDOW_SIZE)
    val_ds   = ToolWearWindowDataset(val_csv, WINDOW_SIZE)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    in_features = train_ds.n_features
    print(f"Window size: {WINDOW_SIZE}, features/timestep: {in_features}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- hyperparameter grid ----
    grid = {
        "d_model":      [16, 32, 64],
        "nhead":        [1, 2],
        "num_layers":   [1, 2],
        "dropout":      [0.2, 0.5],
        "lr":           [1e-4, 3e-4],
        "weight_decay": [1e-4, 1e-3],
    }

    keys = list(grid.keys())
    configs = [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[k] for k in keys))
    ]

    print(f"\nTotal configs to try: {len(configs)}\n")

    best_auc = -1.0
    best_cfg = None
    best_model = None
    best_loss = None

    for i, cfg in enumerate(configs, start=1):
        print(f"=== Config {i}/{len(configs)} ===")
        print(cfg)

        val_loss, val_auc, model = run_single_config(
            cfg, train_loader, val_loader, in_features, device
        )
        print(f" -> Val loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}\n")

        if np.isnan(val_auc):
            continue
        if val_auc > best_auc:
            best_auc = val_auc
            best_cfg = cfg
            best_model = model
            best_loss = val_loss

    print("\n==========================================")
    print(" BEST CONFIG (by validation ROC AUC)")
    print("==========================================")
    print(best_cfg)
    print(f"Best val AUC : {best_auc:.4f}")
    print(f"Best val loss: {best_loss:.4f}")

    # ---- full metrics for best model ----
    _, y_true, y_prob = eval_auc(best_model, val_loader, device)
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nClassification Report (best config):")
    print(classification_report(y_true, y_pred, digits=3))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    try:
        print("PR AUC:", average_precision_score(y_true, y_prob))
    except ValueError:
        print("PR AUC: n/a")

    # save best model
    model_path = PROJECT_ROOT / "models" / "transformer_toolwear_best_hpo.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), model_path)
    print("\nSaved best HPO model to:", model_path)


if __name__ == "__main__":
    main()
