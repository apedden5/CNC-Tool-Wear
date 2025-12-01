import re
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

np.random.seed(42)
tf.random.set_seed(42)

# =====================================
# DATA PREPARATION
# =====================================

def _get_sequence_layout(df, target_col="successful_part", id_col="experiment_id"):
    feature_cols = [c for c in df.columns if c not in [target_col, id_col]]

    pattern = re.compile(r"(.+)_t(\d+)$")
    parsed = []
    for c in feature_cols:
        m = pattern.match(c)
        if not m:
            raise ValueError(f"Invalid column: {c}")
        base, t = m.group(1), int(m.group(2))
        parsed.append((t, base, c))

    parsed_sorted = sorted(parsed)
    ordered_cols = [c for _, _, c in parsed_sorted]

    steps = sorted({t for t, _, _ in parsed_sorted})
    n_steps = len(steps)
    n_features = len(ordered_cols) // n_steps

    return ordered_cols, n_steps, n_features


def _df_to_sequences(df, ordered_cols, n_steps, n_features):
    data = df[ordered_cols].values
    X = data.reshape(len(df), n_steps, n_features)
    y = df["successful_part"].values.astype(int)
    return X, y


def load_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)

    ordered_cols, n_steps, n_features = _get_sequence_layout(train_df)

    X_train, y_train = _df_to_sequences(train_df, ordered_cols, n_steps, n_features)
    X_val, y_val     = _df_to_sequences(val_df,   ordered_cols, n_steps, n_features)
    X_test, y_test   = _df_to_sequences(test_df,  ordered_cols, n_steps, n_features)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(len(X_train), -1)).reshape(len(X_train), n_steps, n_features)
    X_val   = scaler.transform(X_val.reshape(len(X_val), -1)).reshape(len(X_val), n_steps, n_features)
    X_test  = scaler.transform(X_test.reshape(len(X_test), -1)).reshape(len(X_test), n_steps, n_features)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# =====================================
# BASELINE MODEL
# =====================================

def build_baseline_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(64),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def run_baseline_experiment(train_path, val_path, test_path, max_epochs=25):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(train_path, val_path, test_path)

    model = build_baseline_model(X_train.shape[1:])

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=64,
        verbose=0
    )
    train_time = time.time() - start

    y_pred = (model.predict(X_test).ravel() >= 0.5).astype(int)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "confusion": confusion_matrix(y_test, y_pred)
    }

    return {
        "model": model,
        "history": history.history,
        "train_time": train_time,
        "metrics": metrics,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }


# =====================================
# HYPERPARAMETER OPTIMIZATION
# =====================================

def build_lstm_model(input_shape, lstm_units, dropout, learning_rate):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(lstm_units),
        layers.Dropout(dropout),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def run_hpo(X_train, y_train, X_val, y_val, max_epochs=20):

    grid = [
    
    # --- Medium Models (default size) ---
    {"lstm_units": 64,  "dropout": 0.0, "lr": 1e-3},
    {"lstm_units": 64,  "dropout": 0.2, "lr": 1e-3},
    {"lstm_units": 64,  "dropout": 0.3, "lr": 1e-3},

    {"lstm_units": 64,  "dropout": 0.0, "lr": 5e-4},
    {"lstm_units": 64,  "dropout": 0.2, "lr": 5e-4},
    {"lstm_units": 64,  "dropout": 0.3, "lr": 5e-4},

    {"lstm_units": 64,  "dropout": 0.0, "lr": 3e-4},
    {"lstm_units": 64,  "dropout": 0.2, "lr": 3e-4},
    {"lstm_units": 64,  "dropout": 0.3, "lr": 3e-4},

   
    ]

    best = None
    best_acc = -np.inf

    for params in grid:
        print("Testing", params)

        model = build_lstm_model(
            X_train.shape[1:],
            params["lstm_units"],
            params["dropout"],
            params["lr"]
        )

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=64,
            verbose=0
        )

        val_acc = max(history.history["val_accuracy"])

        if val_acc > best_acc:
            best_acc = val_acc
            best = {"params": params, "history": history.history}

    return best


# =====================================
# FINAL RETRAINING USING BEST HPO PARAMS
# =====================================

def retrain_optimized_model(X_train_full, y_train_full, X_test, best_params, max_epochs=25):

    model = build_lstm_model(
        X_train_full.shape[1:],
        best_params["lstm_units"],
        best_params["dropout"],
        best_params["lr"]
    )

    start = time.time()
    history = model.fit(
        X_train_full, y_train_full,
        validation_split=0.2,
        epochs=max_epochs,
        batch_size=64,
        verbose=0
    )
    opt_time = time.time() - start

    y_pred = (model.predict(X_test).ravel() >= 0.5).astype(int)

    return {
        "model": model,
        "history": history.history,
        "train_time": opt_time,
        "preds": y_pred
    }



