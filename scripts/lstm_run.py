import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight


# ---------------------------------------------------------
# LOAD WINDOWED CSV (generic, detects window_len + features)
# ---------------------------------------------------------
def load_window_csv(path, target_col="tool_condition"):
    """
    Loads a windowed CSV where column names follow:
        featureA_t0, featureB_t0, ..., featureA_tN, ..., tool_condition
    Automatically infers:
        - window_len (# timesteps, e.g., 10)
        - num_features (# signals per timestep)
    """

    df = pd.read_csv(path)

    # Extract y labels
    y = df[target_col].values.astype(float)

    # Features
    feature_df = df.drop(columns=[target_col])

    # Determine window length from suffixes
    timestep_suffixes = sorted({col.split("_t")[-1] for col in feature_df.columns})
    window_len = len(timestep_suffixes)

    # Number of features = total columns / timesteps
    total_feature_cols = feature_df.shape[1]
    num_features = total_feature_cols // window_len

    # Reshape into (samples, timesteps, features)
    X = feature_df.values.reshape(len(df), window_len, num_features)

    return X, y, window_len, num_features


# ---------------------------------------------------------
# BUILD LSTM MODEL
# ---------------------------------------------------------
def build_lstm_model(window_len, num_features, units=64, dropout_rate=0.2):

    model = Sequential([
        Input(shape=(window_len, num_features)),

        LSTM(units, return_sequences=True),
        Dropout(dropout_rate),

        LSTM(units // 2),
        Dropout(dropout_rate),

        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")    # Output = probability of worn tool
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2),   # better for imbalance
        metrics=["accuracy"]
    )

    return model


# ---------------------------------------------------------
# TRAIN MODEL WITH CLASS WEIGHTS
# ---------------------------------------------------------
def train_lstm(model, X_train, y_train, X_val, y_val, epochs=40, batch_size=32):

    # Compute class weights => handles imbalance
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    weights = dict(enumerate(weights))
    print("Class weights:", weights)

    # Early stopping for safety
    es = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=weights,
        callbacks=[es],
        verbose=1
    )

    return history


# ---------------------------------------------------------
# PREDICT WITH LOWER THRESHOLD
# ---------------------------------------------------------
def predict_wear(model, X_test, threshold=0.30):
    """
    Returns:
        preds = hard class labels (0/1)
        probs = raw probability output
    """
    probs = model.predict(X_test)
    preds = (probs > threshold).astype(int)
    return preds, probs
