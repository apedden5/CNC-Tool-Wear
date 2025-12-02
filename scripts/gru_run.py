import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import time


# ---------------------------------------------------------
# Load windowed CSVs
# ---------------------------------------------------------
def load_window_csv(path, target_col="successful_part"):
    df = pd.read_csv(path)
    y = df[target_col].values.astype(float)

    feature_df = df.drop(columns=[target_col, "experiment_id"], errors="ignore")

    timestep_idx = sorted({col.split("_t")[-1] for col in feature_df.columns})
    window_len = len(timestep_idx)
    num_features = feature_df.shape[1] // window_len

    X = feature_df.values.reshape(len(df), window_len, num_features)
    return X, y, window_len, num_features


# ---------------------------------------------------------
# Build GRU model
# ---------------------------------------------------------
def build_gru_baseline(window_len, num_features):
    model = Sequential([
        Input(shape=(window_len, num_features)),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2),
        metrics=["accuracy"]
    )
    return model


# ---------------------------------------------------------
# Train GRU baseline model
# ---------------------------------------------------------
def train_gru_baseline(train_path, val_path, epochs=40, batch_size=32):

    X_train, y_train, window_len, num_features = load_window_csv(train_path)
    X_val, y_val, _, _ = load_window_csv(val_path)

    model = build_gru_baseline(window_len, num_features)

    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    weights = dict(enumerate(weights))

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=weights,
        callbacks=[es],
        verbose=1
    )
    train_time = time.time() - start

    return model, history.history, train_time, window_len, num_features, weights


def tune_gru(train_path, val_path,
             learning_rates=[0.001, 0.0005],
             units=[32, 64],
             dropouts=[0.2, 0.3],
             batch_sizes=[16, 32, 64]):

    X_train, y_train, window_len, num_features = load_window_csv(train_path)
    X_val, y_val, _, _ = load_window_csv(val_path)

    best_loss = float("inf")
    best_config = None

    for lr in learning_rates:
        for unit in units:
            for dr in dropouts:
                for bs in batch_sizes:

                    print(f"\nTesting config: LR={lr}, Units={unit}, Dropout={dr}, Batch={bs}")

                    model = Sequential([
                        Input(shape=(window_len, num_features)),
                        GRU(unit, return_sequences=True),
                        Dropout(dr),
                        GRU(unit//2),
                        Dropout(dr),
                        Dense(32, activation="relu"),
                        Dense(1, activation="sigmoid")
                    ])

                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(lr),
                        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2),
                        metrics=["accuracy"]
                    )

                    history = model.fit(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=8,             # keep small for tuning
                        batch_size=bs,        # <---- using batch size here
                        verbose=0
                    )

                    val_loss = history.history["val_loss"][-1]

                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_config = {
                            "learning_rate": lr,
                            "units": unit,
                            "dropout": dr,
                            "batch_size": bs
                        }

    return best_config, best_loss



#
def train_gru_optimized(train_path, val_path, best_config, epochs=40):
   

    
    X_train, y_train, window_len, num_features = load_window_csv(train_path)
    X_val, y_val, _, _ = load_window_csv(val_path)

 
    lr = best_config["learning_rate"]
    units = best_config["units"]
    dr = best_config["dropout"]
    bs = best_config["batch_size"]

    print("\nRetraining GRU Model with Optimized Hyperparameters:")
    print(best_config)

    
    model = Sequential([
        Input(shape=(window_len, num_features)),
        GRU(units, return_sequences=True),
        Dropout(dr),
        GRU(units // 2),
        Dropout(dr),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2),
        metrics=["accuracy"]
    )

   
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    weights = dict(enumerate(weights))

  
    es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=bs,
        class_weight=weights,
        callbacks=[es],
        verbose=1
    )
    train_time = time.time() - start

    return model, history.history, train_time, window_len, num_features











