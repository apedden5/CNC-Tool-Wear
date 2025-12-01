import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc
import time

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


# Defining basic global variables for the dataset and model

DEFAULT_TARGET_COL   = "successful_part"
DEFAULT_WINDOW_SIZE  = 10
NUM_CLASSES          = 2

DEFAULT_BATCH_SIZE   = 64
DEFAULT_EPOCHS_TUNE  = 40
DEFAULT_EPOCHS_FINAL = 60

# Simple baseline hyperparameters for the transformer model, will be overridden during HPO.
BASELINE_PARAMS = {
    "num_heads": 2,
    "ff_dim": 64,
    "num_blocks": 2,
    "dense_units": 32,
    "dropout_rate": 0.2,
    "lr": 1e-3,
}


# Data loading and preprocessing functions
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

# Prepare data for transformer model
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

    # Metadata dictionary
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

# Defining the Transformer Encoder Block
class TransformerEncoderBlock(Layer):
    def __init__(self, num_heads, ff_dim, num_features, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=1)               # key_dim=1 for time series
        self.conv1 = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")       # point-wise FFN
        self.conv2 = Conv1D(filters=num_features, kernel_size=1)                    # project back to num_features
        self.layernorm1 = LayerNormalization(epsilon=1e-6)                          # for attention residual
        self.layernorm2 = LayerNormalization(epsilon=1e-6)                          # for FFN residual
        self.dropout1   = Dropout(rate)                                             # after attention       
        self.dropout2   = Dropout(rate)                                             # after FFN 
    
    # Forward pass
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


# A function to build the Transformer model
def build_transformer(
    time_steps: int,
    num_features: int,
    params: dict | None = None,
):

    # Use baseline params if none provided
    if params is None:
        params = BASELINE_PARAMS

    # Input layer
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
    x = Dense(params["dense_units"], activation="relu")(x)      # relu dense layer
    x = Dropout(params["dropout_rate"])(x)                      # dropout after dense
    outputs = Dense(1, activation="sigmoid")(x)                 # output layer with sigmoid

    model = Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["lr"]),        # Adam optimizer with learning rate
        loss="binary_crossentropy",                                         # binary crossentropy loss
        metrics=["accuracy"],                                               # accuracy metric
    )
    return model


# A function to train the Transformer model
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
    # First, build the model according to the given params
    model = build_transformer(time_steps, num_features, params)

    # Early stopping callback to prevent overfitting
    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    # Train the model and record training time
    start = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[es],
        verbose=verbose,
    )
    end = time.time()

    # Calculate training time in seconds
    train_time = end - start

    return model, history, train_time


# A function to perform hyperparameter optimization using keras-tuner
def _hpo_builder(hp: kt.HyperParameters, time_steps: int, num_features: int) -> Model:
    params = {
        "num_heads":   hp.Choice("num_heads", [2, 4]),                   # number of attention heads
        "ff_dim":      hp.Choice("ff_dim", [64, 128]),                   # feed-forward network dimension
        "num_blocks":  hp.Choice("num_blocks", [1, 2, 3]),               # number of transformer blocks
        "dense_units": hp.Int("dense_units", 32, 128, step=32),          # dense layer units
        "dropout_rate": hp.Choice("dropout_rate", [0.1, 0.2, 0.3]),      # dropout rate
        "lr":          hp.Choice("lr", [1e-3, 5e-4, 2e-4]),              # learning rate
    }
    return build_transformer(time_steps, num_features, params)

# Hyperparameter tuning function
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
    
    # Uses keras-tuner to perform hyperparameter optimization
    keras.backend.clear_session()

    # Define the tuner, using GridSearch
    tuner = kt.GridSearch(
        hypermodel=lambda hp: _hpo_builder(hp, time_steps, num_features),
        objective="val_accuracy",
        max_trials=max_trials,
        directory=directory,
        project_name=project_name,
        overwrite=True,
    )

    # Perform the hyperparameter search
    tuner.search(
        X_train, y_train,
        epochs=epochs_tune,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1,
    )

    # Retrieve the best hyperparameters
    best_hp = tuner.get_best_hyperparameters(1)[0]

    # Print the best hyperparameters
    print("\nBest hyperparameters:")
    for name in ["num_heads", "ff_dim", "num_blocks",
                 "dense_units", "dropout_rate", "lr"]:
        print(f"  {name}: {best_hp.get(name)}")

    return best_hp, tuner

# A function to convert keras-tuner HyperParameters to plain dict for model building
def hp_to_params(best_hp: kt.HyperParameters) -> dict:
    return {
        "num_heads":   best_hp.get("num_heads"),
        "ff_dim":      best_hp.get("ff_dim"),
        "num_blocks":  best_hp.get("num_blocks"),
        "dense_units": best_hp.get("dense_units"),
        "dropout_rate": best_hp.get("dropout_rate"),
        "lr":          best_hp.get("lr"),
    }


# A function to evaluate the model and plot metrics
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

    # Plot training & validation loss
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_prob = model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    f1 = f1_score(y_test, y_pred)               # F1-score
    cm = confusion_matrix(y_test, y_pred)       # Confusion matrix
    fpr, tpr, _ = roc_curve(y_test, y_prob)     # ROC curve
    roc_auc = auc(fpr, tpr)                     # AUC

    # Output metrics
    print(f"\nTest loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    print(f"F1-score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    print("Confusion matrix:\n", cm)

    return {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "cm": cm,
    }
