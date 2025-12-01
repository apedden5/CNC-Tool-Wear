import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid OpenMP runtime clash on Windows

import math
from pathlib import Path
import random
import json
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc

# ----------------- Global config -----------------

TARGET_COL   = "successful_part"
WINDOW_SIZE  = 10
NUM_CLASSES  = 2

BATCH_SIZE   = 96          # slightly larger batch for smoother gradients
WEIGHT_DECAY = 0.0         # keep simple / explainable

EPOCHS_HPO   = 6           # if you ever re-run HPO
EPOCHS_FINAL = 80          # early stopping will usually stop earlier

SEED         = 42


# ----------------- Paths -----------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


BASE_DIR = _project_root()
WIN_DIR  = BASE_DIR / "data" / "data_windowed_csv"

TRAIN_CSV = WIN_DIR / f"train_windows_w{WINDOW_SIZE}.csv"
VAL_CSV   = WIN_DIR / f"val_windows_w{WINDOW_SIZE}.csv"
TEST_CSV  = WIN_DIR / f"test_windows_w{WINDOW_SIZE}.csv"


# ----------------- Reproducibility -----------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------- Dataset -----------------

class WindowedCNCDataset(Dataset):
    """
    Loads flattened window CSVs and reconstructs (N, W, F) tensors.
    Ignores experiment_id so we don't leak data.
    """
    def __init__(self, csv_path: Path, target_col: str = TARGET_COL, window_size: int = WINDOW_SIZE):
        df = pd.read_csv(csv_path)

        if target_col not in df.columns:
            raise ValueError(f"{target_col} not found in {csv_path}")

        non_feat_cols   = {target_col, "experiment_id"}
        flat_feat_cols  = [c for c in df.columns if c not in non_feat_cols]

        base_feats   = sorted({col.rsplit("_t", 1)[0] for col in flat_feat_cols})
        time_indices = sorted({int(col.rsplit("_t", 1)[1]) for col in flat_feat_cols})

        W = len(time_indices)
        if W != window_size:
            raise ValueError(f"Window size mismatch: got {W}, expected {window_size}")

        self.base_feats   = base_feats
        self.window_size  = window_size
        self.num_features = len(base_feats)

        X_list = []
        for t in time_indices:
            cols_t = [f"{feat}_t{t}" for feat in base_feats]
            X_t = df[cols_t].values.astype(np.float32)  # (N, F)
            X_list.append(X_t)

        X = np.stack(X_list, axis=1)                    # (N, W, F)
        y = df[target_col].values.astype(np.int64)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------- Transformer pieces (lab-style) -----------------

class TransformerEncoderBlock(nn.Module):
    """
    Keras-style encoder block:

    attn = MultiHeadAttention(...)
    out1 = LayerNorm(x + Dropout(attn(x)))
    ffn  = Conv1D(1x1) -> ReLU -> Conv1D(1x1)
    out2 = LayerNorm(out1 + Dropout(ffn(out1)))
    """
    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.att = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # (batch, time, dim)
        )

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=d_model, kernel_size=1)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1   = nn.Dropout(dropout)
        self.dropout2   = nn.Dropout(dropout)

    def forward(self, inputs):
        # inputs: (batch, time, d_model)

        # Self-attention + residual + norm
        attn_output, _ = self.att(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)

        # Conv1D FFN (1x1 convs) + residual + norm
        x = out1.transpose(1, 2)  # (batch, d_model, time)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)     # back to (batch, time, d_model)

        x = self.dropout2(x)
        out2 = self.layernorm2(out1 + x)
        return out2


class SimpleTransformerClassifier(nn.Module):
    """
    Stacks N encoder blocks, then:

      Flatten -> Dense(dense_units) -> (optional Dropout) -> Dense(num_classes)

    Mirrors the Keras lab model (no CLS token, just flatten).
    """
    def __init__(
        self,
        input_dim,
        window_size,
        num_classes,
        num_blocks=4,
        num_heads=4,
        d_model=32,
        ff_dim=64,
        dense_units=32,
        use_dense_dropout=False,
        dense_dropout_rate=0.3,
    ):
        super().__init__()

        self.window_size = window_size
        self.d_model     = d_model

        self.input_proj = nn.Linear(input_dim, d_model)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=0.1,
                )
                for _ in range(num_blocks)
            ]
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(window_size * d_model, dense_units)

        if use_dense_dropout:
            self.dense_dropout = nn.Dropout(dense_dropout_rate)
        else:
            self.dense_dropout = nn.Identity()

        self.out = nn.Linear(dense_units, num_classes)

    def forward(self, x):
        # x: (batch, time, input_dim)
        x = self.input_proj(x)   # (batch, time, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.flatten(x)      # (batch, time * d_model)
        x = F.relu(self.fc(x))
        x = self.dense_dropout(x)
        logits = self.out(x)
        return logits


# ----------------- Training helpers -----------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * y_batch.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

    return total_loss / total, correct / total


# ----------------- Data loader helper -----------------

def prepare_data():
    """Create train/val/test DataLoaders and return input_dim."""
    train_ds = WindowedCNCDataset(TRAIN_CSV)
    val_ds   = WindowedCNCDataset(VAL_CSV)
    test_ds  = WindowedCNCDataset(TEST_CSV)

    input_dim = train_ds.num_features

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, input_dim


# ----------------- (Optional) HPO loop -----------------
# You can ignore this if you've already run HPO and saved transformer_best_config.json.

def run_hpo(train_loader, val_loader, input_dim, device):
    """
    Smaller, explainable grid search:
      - d_model       ∈ {32, 64}
      - dense_units   ∈ {32, 64}
      - use_dropout   ∈ {False, True}
      - lr            ∈ {1e-3, 5e-4}
    We fix:
      - num_heads = 4
      - ff_dim    = 2 * d_model
    """
    criterion = nn.CrossEntropyLoss()

    d_model_values    = [32, 64]
    dense_unit_values = [32, 64]
    use_dropout_vals  = [False, True]
    lr_values         = [1e-3, 5e-4]

    best_val_loss = float("inf")
    best_config   = None

    total_trials = (
        len(d_model_values)
        * len(dense_unit_values)
        * len(use_dropout_vals)
        * len(lr_values)
    )
    trial = 0

    for d_model in d_model_values:
        ff_dim = 2 * d_model  # simple rule: FFN size = 2×embedding
        for dense_units in dense_unit_values:
            for use_dropout in use_dropout_vals:
                for lr in lr_values:
                    trial += 1
                    print(
                        f"\n=== HPO trial {trial}/{total_trials} ===\n"
                        f"d_model={d_model}, ff_dim={ff_dim}, "
                        f"dense_units={dense_units}, use_dropout={use_dropout}, lr={lr}"
                    )

                    model = SimpleTransformerClassifier(
                        input_dim=input_dim,
                        window_size=WINDOW_SIZE,
                        num_classes=NUM_CLASSES,
                        num_blocks=4,
                        num_heads=4,
                        d_model=d_model,
                        ff_dim=ff_dim,
                        dense_units=dense_units,
                        use_dense_dropout=use_dropout,
                        dense_dropout_rate=0.3,
                    ).to(device)

                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=lr,
                        weight_decay=WEIGHT_DECAY,
                    )

                    best_trial_val_loss = float("inf")

                    for epoch in range(1, EPOCHS_HPO + 1):
                        train_loss, train_acc = train_one_epoch(
                            model, train_loader, criterion, optimizer, device
                        )
                        val_loss, val_acc = evaluate(
                            model, val_loader, criterion, device
                        )

                        best_trial_val_loss = min(best_trial_val_loss, val_loss)

                        if epoch in {1, EPOCHS_HPO}:
                            print(
                                f"  Epoch {epoch:02d}: "
                                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                                f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}"
                            )

                    print(f"  -> Best val_loss for this config: {best_trial_val_loss:.4f}")

                    if best_trial_val_loss < best_val_loss:
                        best_val_loss = best_trial_val_loss
                        best_config = {
                            "d_model": d_model,
                            "ff_dim": ff_dim,
                            "dense_units": dense_units,
                            "use_dropout": use_dropout,
                            "lr": lr,
                            "num_heads": 4,
                        }

    print("\n=== HPO finished ===")
    print(f"Best config: {best_config}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    return best_config


def run_hpo_only(device):
    """Run HPO once and save the best config to JSON."""
    train_loader, val_loader, test_loader, input_dim = prepare_data()
    best_cfg = run_hpo(train_loader, val_loader, input_dim, device)

    cfg_path = BASE_DIR / "transformer_best_config.json"
    with open(cfg_path, "w") as f:
        json.dump(best_cfg, f, indent=4)
    print(f"Saved best config to {cfg_path}")


# ----------------- Final training (with metrics & plots) -----------------

def train_final_only(device, cfg=None):
    """
    Train the final model using a given config (or load from JSON),
    return history + test metrics, and plot loss/accuracy curves,
    confusion matrix, and ROC curve.
    """
    # Load config from JSON if not provided
    if cfg is None:
        cfg_path = BASE_DIR / "transformer_best_config.json"
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        print(f"Loaded best config from {cfg_path}: {cfg}")
    else:
        print(f"Using provided config: {cfg}")

    train_loader, val_loader, test_loader, input_dim = prepare_data()

    # Build model with best hyperparameters
    model = SimpleTransformerClassifier(
        input_dim=input_dim,
        window_size=WINDOW_SIZE,
        num_classes=NUM_CLASSES,
        num_blocks=4,
        num_heads=cfg["num_heads"],
        d_model=cfg["d_model"],
        ff_dim=cfg["ff_dim"],
        dense_units=cfg["dense_units"],
        use_dense_dropout=cfg["use_dropout"],
        dense_dropout_rate=0.3,
    ).to(device)

    # Slightly smaller LR than HPO for smoother convergence
    base_lr = cfg["lr"] * 0.5
    print(f"Using final training LR = {base_lr}")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base_lr,
        weight_decay=WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-5,
    )

    train_losses, val_losses = [], []
    train_accs, val_accs     = [], []

    best_val_loss = float("inf")
    best_state    = None
    patience_es   = 10
    epochs_no_improve = 0

    # -------- Training loop --------
    for epoch in range(1, EPOCHS_FINAL + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{EPOCHS_FINAL} "
            f"- train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}"
        )

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_es:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # -------- Test metrics (including F1, confusion matrix, ROC) --------
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    # Collect predictions & probabilities for detailed metrics
    all_labels = []
    all_preds  = []
    all_probs  = []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)[:, 1]  # prob of class 1
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(y_batch.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    print(f"\nFinal Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    print(f"Test F1-score: {f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    print(f"ROC AUC: {roc_auc:.4f}")

    history = {
        "train_loss": train_losses,
        "val_loss":   val_losses,
        "train_acc":  train_accs,
        "val_acc":    val_accs,
    }
    test_metrics = {
        "test_loss": test_loss,
        "test_acc":  test_acc,
        "f1":        f1,
        "confusion_matrix": cm.tolist(),
        "roc_auc":   roc_auc,
    }

    # -------- Smoothing for nicer plots --------
    def moving_avg(x, k=3):
        if len(x) < k:
            return x
        out = []
        for i in range(len(x)):
            start = max(0, i - k + 1)
            out.append(sum(x[start:i+1]) / (i - start + 1))
        return out

    sm_train_losses = moving_avg(train_losses, k=3)
    sm_val_losses   = moving_avg(val_losses,   k=3)
    sm_train_accs   = moving_avg(train_accs,   k=3)
    sm_val_accs     = moving_avg(val_accs,     k=3)

    epochs = range(1, len(train_losses) + 1)

    # -------- Loss plot --------
    plt.figure()
    plt.plot(epochs, sm_train_losses, label="Train Loss (smoothed)")
    plt.plot(epochs, sm_val_losses,   label="Val Loss (smoothed)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Transformer – Successful Part (Train vs Val Loss)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "transformer_final_loss.png", dpi=200)
    plt.show()

    # -------- Accuracy plot --------
    plt.figure()
    plt.plot(epochs, sm_train_accs, label="Train Acc (smoothed)")
    plt.plot(epochs, sm_val_accs,   label="Val Acc (smoothed)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Transformer – Successful Part (Train vs Val Accuracy)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "transformer_final_acc.png", dpi=200)
    plt.show()

    # -------- Confusion matrix plot --------
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)")
    plt.colorbar(im)
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, [0, 1])
    plt.yticks(tick_marks, [0, 1])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    # annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(BASE_DIR / "transformer_confusion_matrix.png", dpi=200)
    plt.show()

    # -------- ROC curve plot --------
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve – Successful Part (Test Set)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "transformer_roc_curve.png", dpi=200)
    plt.show()

    # Save final model weights
    torch.save(model.state_dict(), BASE_DIR / "transformer_final.pt")
    print(f"Saved final model weights to {BASE_DIR / 'transformer_final.pt'}")

    return history, test_metrics


# ----------------- Main -----------------

def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Set to "hpo" if you ever want to re-run hyperparameter search.
    MODE = "final"   # "hpo" or "final"

    if MODE == "hpo":
        run_hpo_only(device)
    elif MODE == "final":
        history, test_metrics = train_final_only(device)
        print("Test metrics:", test_metrics)
    else:
        raise ValueError(f"Unknown MODE: {MODE}")


if __name__ == "__main__":
    main()
