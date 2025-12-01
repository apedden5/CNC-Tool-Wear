import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.layers import (
    Layer,
    MultiHeadAttention,
    LayerNormalization,
    Dropout,
    Conv1D,
    Input,
    Flatten,
    Dense,
)
from keras.models import Model
from keras.callbacks import EarlyStopping

import keras_tuner as kt
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc


# ----------- basic config -----------

DEFAULT_TARGET_COL   = "successful_part"
DEFAULT_WINDOW_SIZE  = 10
NUM_CLASSES          = 2

DEFAULT_BATCH_SIZE   = 64
DEFAULT_EPOCHS_TUNE  = 40
DEFAULT_EPOCHS_FINAL = 60

# simple baseline (also in HPO search space)
BASELINE_PARAMS = {
    "num_heads": 2,
    "ff_dim": 64,
    "num_blocks": 2,
    "dense_units": 32,
    "dropout_rate": 0.2,
    "lr": 1e-3,
}


# ----------- data helpers -----------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _window_paths(window_size: int, base_dir: Path | None = None) -> dict:
    if base_dir is None:
        base_dir = _project_root()
    win_dir = base_dir / "data" / "data_windowed_csv"
    return {
        "base_dir": base_dir,
        "train": win_dir / f"train_windows_w{window_size}.csv",
        "val":   win_dir / f"val_windows_w{window_size}.csv",
        "test":  win_dir / f"test_windows_w{window_size}.csv",
    }


def _load_windowed(csv_path: Path, target_col: str):
    df = pd.read_csv(csv_path)

    non_feat = {target_col, "experiment_id"}
    flat_cols = [c for c in df.columns if c not in non_feat]

    base_feats   = sorted({c.rsplit("_t", 1)[0] for c in flat_cols})
    time_indices = sorted({int(c.rsplit("_t", 1)[1]) for c in flat_cols})

    X_list = []
    for t in time_indices:
        cols_t = [f"{f}_t{t}" for f in base_feats]
        X_t = df[cols_t].values.astype("float32")
        X_list.append(X_t)

    X = np.stack(X_list, axis=1)  # (N, W, F)
    y = df[target_col].values.astype("float32")
    return X, y


def load_transformer_data(
    window_size: int = DEFAULT_WINDOW_SIZE,
    target_col: str = DEFAULT_TARGET_COL,
    base_dir: Path | None = None,
):
    paths = _window_paths(window_size, base_dir)

    X_train, y_train = _load_windowed(paths["train"], target_col)
    X_val,   y_val   = _load_windowed(paths["val"], target_col)
    X_test,  y_test  = _load_windowed(paths["test"], target_col)

    time_steps   = X_train.shape[1]
    num_features = X_train.shape[2]

    meta = {
        "time_steps": time_steps,
        "num_features": num_features,
        "paths": paths,
        "target_col": target_col,
        "window_size": window_size,
    }

    print("time_steps:", time_steps,
          "num_features:", num_features,
          "num_classes:", NUM_CLASSES)

    return X_train, y_train, X_val, y_val, X_test, y_test, meta


# ----------- transformer block (same idea as your lab) -----------

class TransformerEncoderBlock(Layer):
    def __init__(self, num_heads, ff_dim, num_features, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=1)
        self.conv1 = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")
        self.conv2 = Conv1D(filters=num_features, kernel_size=1)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1   = Dropout(rate)
        self.dropout2   = Dropout(rate)

    def call(self, inputs, training=False):
        # self-attention + residual
        attn = self.att(inputs, inputs)
        attn = self.dropout1(attn, training=training)
        out1 = self.layernorm1(inputs + attn)

        # 1x1 conv FFN + residual
        ffn = self.conv1(out1)
        ffn = self.conv2(ffn)
        ffn = self.dropout2(ffn, training=training)
        return self.layernorm2(out1 + ffn)


# ----------- build model from params -----------

def build_transformer(
    time_steps: int,
    num_features: int,
    params: dict | None = None,
):
    """
    params keys: num_heads, ff_dim, num_blocks, dense_units, dropout_rate, lr
    If params is None -> BASELINE_PARAMS.
    """
    if params is None:
        params = BASELINE_PARAMS

    inputs = Input(shape=(time_steps, num_features))
    x = inputs

    for _ in range(params["num_blocks"]):
        x = TransformerEncoderBlock(
            num_heads=params["num_heads"],
            ff_dim=params["ff_dim"],
            num_features=num_features,
            rate=params["dropout_rate"],
        )(x)

    x = Flatten()(x)
    x = Dense(params["dense_units"], activation="relu")(x)
    x = Dropout(params["dropout_rate"])(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["lr"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ----------- generic training (no plots) -----------

def train_transformer(
    X_train,
    y_train,
    X_val,
    y_val,
    time_steps: int,
    num_features: int,
    params: dict | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS_FINAL,
    patience: int = 10,
    verbose: int = 1,
):
    """
    Generic training:
      - baseline: call with params=None
      - tuned:    call with params from HPO
    Returns: model, history
    """
    model = build_transformer(time_steps, num_features, params)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[es],
        verbose=verbose,
    )

    return model, history


# ----------- HPO (minimal) -----------

def _hpo_builder(hp: kt.HyperParameters, time_steps: int, num_features: int) -> Model:
    params = {
        "num_heads":   hp.Choice("num_heads", [2, 4]),
        "ff_dim":      hp.Choice("ff_dim", [64, 128]),
        "num_blocks":  hp.Choice("num_blocks", [1, 2, 3]),
        "dense_units": hp.Int("dense_units", 32, 128, step=32),
        "dropout_rate": hp.Choice("dropout_rate", [0.1, 0.2, 0.3]),
        "lr":          hp.Choice("lr", [1e-3, 5e-4, 2e-4]),
    }
    return build_transformer(time_steps, num_features, params)


def tune_transformer(
    X_train,
    y_train,
    X_val,
    y_val,
    time_steps: int,
    num_features: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs_tune: int = DEFAULT_EPOCHS_TUNE,
    max_trials: int = 25,
    directory: str = "transformer_hpo",
    project_name: str = "transformer_search",
):
    """
    keras-tuner GridSearch around the same hyperparams as BASELINE_PARAMS.
    Returns: best_hp, tuner
    """
    keras.backend.clear_session()

    tuner = kt.GridSearch(
        hypermodel=lambda hp: _hpo_builder(hp, time_steps, num_features),
        objective="val_accuracy",
        max_trials=max_trials,
        directory=directory,
        project_name=project_name,
        overwrite=True,
    )

    tuner.search(
        X_train, y_train,
        epochs=epochs_tune,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]

    print("\nBest hyperparameters:")
    for name in ["num_heads", "ff_dim", "num_blocks",
                 "dense_units", "dropout_rate", "lr"]:
        print(f"  {name}: {best_hp.get(name)}")

    return best_hp, tuner


def hp_to_params(best_hp: kt.HyperParameters) -> dict:
    """Convert keras-tuner HyperParameters -> plain dict usable by train_transformer."""
    return {
        "num_heads":   best_hp.get("num_heads"),
        "ff_dim":      best_hp.get("ff_dim"),
        "num_blocks":  best_hp.get("num_blocks"),
        "dense_units": best_hp.get("dense_units"),
        "dropout_rate": best_hp.get("dropout_rate"),
        "lr":          best_hp.get("lr"),
    }


# ----------- optional evaluation + plots (only if you call it) -----------

def evaluate_and_plot(
    model: Model,
    history,
    X_test,
    y_test,
    base_dir: Path | None = None,
    prefix: str = "transformer",
):
    if base_dir is None:
        base_dir = _project_root()

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_prob = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\nTest loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    print(f"F1-score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    print("Confusion matrix:\n", cm)

    # (you can keep / delete plots as you like)
    # ...
    return {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "cm": cm,
    }
