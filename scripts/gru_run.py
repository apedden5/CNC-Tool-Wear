import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
import time
import os, json, pickle


def load_window_csv(path, target_col="successful_part"):

    df = pd.read_csv(path)
    y = df[target_col].values.astype(float)

    feature_df = df.drop(columns=[target_col, "experiment_id"], errors="ignore")

    timestep_idx = sorted({col.split("_t")[-1] for col in feature_df.columns})
    window_len = len(timestep_idx)
    num_features = feature_df.shape[1] // window_len

    X = feature_df.values.reshape(len(df), window_len, num_features)

    return X, y, window_len, num_features





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




def save_baseline(model, history, train_time, window_len, num_features, weights, save_dir="saved_baseline"):

    os.makedirs(save_dir, exist_ok=True)

    # Save model
    model.save(os.path.join(save_dir, "baseline_model.h5"))

    # Save training history
    with open(os.path.join(save_dir, "history.pkl"), "wb") as f:
        pickle.dump(history, f)

    # Save metadata
    meta = {
        "train_time": train_time,
        "window_len": window_len,
        "num_features": num_features,
        "class_weights": weights
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)




def load_saved_baseline(save_dir="saved_baseline"):
    model = load_model(os.path.join(save_dir, "baseline_model.h5"))

    with open(os.path.join(save_dir, "history.pkl"), "rb") as f:
        history = pickle.load(f)

    with open(os.path.join(save_dir, "meta.json"), "r") as f:
        meta = json.load(f)

    return {
        "model": model,
        "history": history,
        "train_time": meta["train_time"],
        "class_weights": meta["class_weights"],
        "window_len": meta["window_len"],
        "num_features": meta["num_features"]
    }




if __name__ == "__main__":
    print(" Training GRU baseline model...")

    # Hardcode your paths ONCE here
    train_path = "data/data_windowed_csv/train_windows_w10.csv"
    val_path   = "data/data_windowed_csv/val_windows_w10.csv"

    model, history, train_time, window_len, num_features, weights = train_gru_baseline(
        train_path, val_path
    )

    save_baseline(model, history, train_time, window_len, num_features, weights)

    print("Baseline GRU training complete and saved.")
