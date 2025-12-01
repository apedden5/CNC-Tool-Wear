# lstm_successful_part_w10.py

import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def _get_sequence_layout(train_df, target_col="successful_part", id_col="experiment_id"):
    """
    From the training DataFrame, infer:
    - Which columns are features (and in what order)
    - Number of time steps and features per step
    Returns: ordered_cols, n_steps, n_features
    """
    feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]

    pattern = re.compile(r"(.+)_t(\d+)$")
    parsed = []
    for c in feature_cols:
        m = pattern.match(c)
        if not m:
            raise ValueError(f"Column '{c}' does not match the pattern 'name_tk'")
        base, t = m.group(1), int(m.group(2))
        parsed.append((t, base, c))

    # Sort by time step, then by base name
    parsed_sorted = sorted(parsed)  # (t, base, col)
    ordered_cols = [c for _, _, c in parsed_sorted]

    steps = sorted({t for t, _, _ in parsed_sorted})
    n_steps = len(steps)
    n_features = len(ordered_cols) // n_steps

    return ordered_cols, n_steps, n_features


def _df_to_sequences(df, ordered_cols, n_steps, n_features, target_col="successful_part"):
    """
    Convert a DataFrame into (X, y) where:
    - X: (samples, n_steps, n_features)
    - y: (samples,)
    """
    data = df[ordered_cols].values
    X = data.reshape(len(df), n_steps, n_features)
    y = df[target_col].values.astype("int32")
    return X, y


def load_and_prepare_datasets(
    train_path,
    val_path,
    test_path,
    target_col="successful_part",
    id_col="experiment_id",
):
    """
    Load CSVs, reshape into LSTM-ready sequences, and scale features.
    Returns:
      (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    ordered_cols, n_steps, n_features = _get_sequence_layout(
        train_df, target_col=target_col, id_col=id_col
    )

    X_train, y_train = _df_to_sequences(
        train_df, ordered_cols, n_steps, n_features, target_col=target_col
    )
    X_val, y_val = _df_to_sequences(
        val_df, ordered_cols, n_steps, n_features, target_col=target_col
    )
    X_test, y_test = _df_to_sequences(
        test_df, ordered_cols, n_steps, n_features, target_col=target_col
    )

    # Scale features (flatten time dimension, then reshape back)
    scaler = StandardScaler()
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]
    n_test = X_test.shape[0]

    X_train_2d = X_train.reshape(n_train, -1)
    X_val_2d = X_val.reshape(n_val, -1)
    X_test_2d = X_test.reshape(n_test, -1)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(
        n_train, n_steps, n_features
    )
    X_val_scaled = scaler.transform(X_val_2d).reshape(n_val, n_steps, n_features)
    X_test_scaled = scaler.transform(X_test_2d).reshape(n_test, n_steps, n_features)

    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test), scaler


def build_lstm_model(
    input_shape,
    lstm_units=64,
    dropout=0.3,
    learning_rate=1e-3,
):
    """
    Build a simple LSTM model for binary classification.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.LSTM(lstm_units, return_sequences=False))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    optimizer = optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def hyperparameter_search(
    X_train,
    y_train,
    X_val,
    y_val,
    max_epochs=25,
):
    """
    Very simple manual hyperparameter search (grid search) over a few configs.
    Returns: best_model, best_config, history_per_config
    """
    # You can expand this grid; kept small so it's practical to run
    param_grid = [
        {"lstm_units": 32, "dropout": 0.0, "learning_rate": 1e-3, "batch_size": 64},
        {"lstm_units": 64, "dropout": 0.3, "learning_rate": 1e-3, "batch_size": 64},
        {"lstm_units": 64, "dropout": 0.3, "learning_rate": 3e-4, "batch_size": 64},
        {"lstm_units": 128, "dropout": 0.3, "learning_rate": 1e-3, "batch_size": 64},
    ]

    best_val_acc = -np.inf
    best_config = None
    best_model = None
    history_per_config = {}

    input_shape = X_train.shape[1:]

    for i, params in enumerate(param_grid):
        print(f"\n=== Config {i + 1}/{len(param_grid)}: {params} ===")

        model = build_lstm_model(
            input_shape=input_shape,
            lstm_units=params["lstm_units"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
        )

        es = callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=params["batch_size"],
            callbacks=[es],
            verbose=0,  # set to 1 if you want per-epoch logs
        )

        history_per_config[str(params)] = history.history

        # Use best validation accuracy from training as selection metric
        val_acc_list = history.history.get("val_accuracy", [])
        if len(val_acc_list) == 0:
            val_acc = 0.0
        else:
            val_acc = max(val_acc_list)

        print(f"Best val_accuracy for this config: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_config = params
            best_model = model

    print("\n=== Hyperparameter search finished ===")
    print(f"Best config: {best_config}")
    print(f"Best val_accuracy: {best_val_acc:.4f}")

    return best_model, best_config, history_per_config


def run_lstm_experiment(
    train_path="train_windows_w10.csv",
    val_path="val_windows_w10.csv",
    test_path="test_windows_w10.csv",
    max_epochs=25,
):
    """
    High-level function you call from your notebook.
    Loads data, runs hyperparameter search, and prints out test-set results.
    """
    print("Loading and preparing datasets...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = load_and_prepare_datasets(
        train_path, val_path, test_path
    )

    print("Shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    best_model, best_config, history_per_config = hyperparameter_search(
        X_train, y_train, X_val, y_val, max_epochs=max_epochs
    )

    print("\nEvaluating best model on TEST set...")
    test_probs = best_model.predict(X_test).ravel()
    test_preds = (test_probs >= 0.5).astype("int32")

    test_loss, test_acc, test_auc = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test AUC: {test_auc:.4f}")

    print("\nClassification report (test):")
    print(classification_report(y_test, test_preds, digits=4))

    print("Confusion matrix (test):")
    print(confusion_matrix(y_test, test_preds))

    # Return things if you want to inspect them further in your notebook
    return {
        "model": best_model,
        "best_config": best_config,
        "history_per_config": history_per_config,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


# Optional: so you can run this file directly as a script if you want
if __name__ == "__main__":
    run_lstm_experiment()

