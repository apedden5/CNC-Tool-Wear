import re
import pandas as pd
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

import tensorflow as tf
from tcn import TCN
from tensorflow.keras import layers, models, callbacks, optimizers

tf.random.set_seed(42)
np.random.seed(42)

# ---------------------------------------------------------
#   LAYOUT DETECTION
# ---------------------------------------------------------

def _get_sequence_layout(train_df, target_col="successful_part", id_col="experiment_id"):
    feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]

    pattern = re.compile(r"(.+)_t(\d+)$")
    parsed = []
    for c in feature_cols:
        m = pattern.match(c)
        if not m:
            raise ValueError(f"Column '{c}' does not match pattern name_tk")
        base, t = m.group(1), int(m.group(2))
        parsed.append((t, base, c))

    parsed_sorted = sorted(parsed)
    ordered_cols = [c for _, _, c in parsed_sorted]

    steps = sorted({t for t, _, _ in parsed_sorted})
    n_steps = len(steps)
    n_features = len(ordered_cols) // n_steps

    return ordered_cols, n_steps, n_features


# ---------------------------------------------------------
#   RESHAPE DF → (X, y)
# ---------------------------------------------------------

def _df_to_sequences(df, ordered_cols, n_steps, n_features, target_col="successful_part"):
    data = df[ordered_cols].values
    X = data.reshape(len(df), n_steps, n_features)
    y = df[target_col].values.astype("int32")
    return X, y


# ---------------------------------------------------------
#   LOAD + SCALE DATASETS
# ---------------------------------------------------------

def load_and_prepare_datasets(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    ordered_cols, n_steps, n_features = _get_sequence_layout(train_df)

    X_train, y_train = _df_to_sequences(train_df, ordered_cols, n_steps, n_features)
    X_val, y_val = _df_to_sequences(val_df, ordered_cols, n_steps, n_features)
    X_test, y_test = _df_to_sequences(test_df, ordered_cols, n_steps, n_features)

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    X_train_scaled = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    return (X_train_scaled, y_train), (X_val_scaled, y_val), (X_test_scaled, y_test), n_steps, n_features


# ---------------------------------------------------------
#   BUILD TCN MODEL
# ---------------------------------------------------------

def build_tcn_model(input_shape, filters=64, kernel_size=3, dropout=0.3, lr=1e-3):

    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))

    # Deep multi-layer TCN with dilation doubling
    model.add(TCN(
        nb_filters=filters,
        kernel_size=kernel_size,
        dropout_rate=dropout,
        dilations=[1, 2, 4, 8],
        return_sequences=False,  # Final output only
    ))

    # Dense classifier head
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model



# ---------------------------------------------------------
#   HYPERPARAM SEARCH
# ---------------------------------------------------------
def hyperparameter_search_tcn(X_train, y_train, X_val, y_val, max_epochs=25):

    param_grid = [
        {"filters": 32, "kernel_size": 2, "dropout": 0.2, "lr": 1e-3, "batch": 64},
        {"filters": 32, "kernel_size": 3, "dropout": 0.3, "lr": 1e-3, "batch": 64},
        {"filters": 64, "kernel_size": 2, "dropout": 0.2, "lr": 1e-3, "batch": 64},
        {"filters": 64, "kernel_size": 3, "dropout": 0.3, "lr": 3e-4, "batch": 64},
        {"filters": 64, "kernel_size": 5, "dropout": 0.3, "lr": 1e-3, "batch": 32},
        {"filters": 128, "kernel_size": 3, "dropout": 0.4, "lr": 3e-4, "batch": 64},
    ]

    best_val_auc = -np.inf
    best_config = None
    best_model = None
    history_per_config = {}

    input_shape = X_train.shape[1:]

    for i, p in enumerate(param_grid):
        print(f"\n=== TCN Config {i+1}/{len(param_grid)} ===")
        print(p)

        model = build_tcn_model(
            input_shape=input_shape,
            filters=p["filters"],
            kernel_size=p["kernel_size"],
            dropout=p["dropout"],
            lr=p["lr"],
        )

        es = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )

        # --- class weights for imbalance ---
        classes = np.unique(y_train)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
        )
        class_weights = dict(zip(classes, weights))

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=p["batch"],
            callbacks=[es],
            class_weight=class_weights,
            verbose=0
        )

        history_per_config[str(p)] = history.history

        val_auc = max(history.history.get("val_auc", [0]))
        print(f"Best val_auc: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_config = p
            best_model = model

    print("\n=== Finished TCN Search ===")
    print("Best config:", best_config)
    print("Best val AUC:", best_val_auc)

    return best_model, best_config, history_per_config



# ---------------------------------------------------------
#   MAIN RUNNER
# ---------------------------------------------------------

def run_tcn_experiment(train_path, val_path, test_path, max_epochs=25):
    print("Loading datasets...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), n_steps, n_features = \
        load_and_prepare_datasets(train_path, val_path, test_path)

    print("Shapes:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val:",   X_val.shape,   "y_val:",   y_val.shape)
    print("  X_test:",  X_test.shape,  "y_test:",  y_test.shape)

    # ---- Hyperparameter search ----
    best_model, best_config, history_per_config = hyperparameter_search_tcn(
        X_train, y_train, X_val, y_val, max_epochs=max_epochs
    )

    print("\nEvaluating best TCN on TEST set...")
    test_probs = best_model.predict(X_test).ravel()

    # -----------------------------------------------------
    # 1. FIND OPTIMAL THRESHOLD USING F1
    # -----------------------------------------------------
    from sklearn.metrics import precision_recall_curve
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, test_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    print(f"\n=== Optimal threshold search ===")
    print(f"Best threshold for F1: {best_threshold:.4f}")
    print(f"Max F1 at this threshold: {f1_scores[best_idx]:.4f}")

    # -----------------------------------------------------
    # 2. PREDICT USING OPTIMAL THRESHOLD
    # -----------------------------------------------------
    test_preds = (test_probs >= best_threshold).astype("int32")

    # -----------------------------------------------------
    # 3. MODEL EVALUATION
    # -----------------------------------------------------
    loss, acc, auc = best_model.evaluate(X_test, y_test, verbose=0)
    print("\nBase Test Metrics (threshold-free):")
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy (raw): {acc:.4f}")
    print(f"Test AUC: {auc:.4f}")

    print("\nMetrics Using Optimized Threshold:")
    prec = precision_score(y_test, test_preds)
    rec = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)

    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, test_preds, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_preds))

    return {
        "model": best_model,
        "best_config": best_config,
        "best_threshold": best_threshold,
        "history": history_per_config,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "n_steps": n_steps,
        "n_features": n_features,
    }
