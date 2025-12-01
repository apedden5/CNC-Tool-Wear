import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight


# ---------------------------------------------------------
# LOAD WINDOWED CSV WITH NEW TARGET "successful_part"
# ---------------------------------------------------------
def load_window_csv(path, target_col="successful_part"):

    df = pd.read_csv(path)

    # Extract labels
    y = df[target_col].values.astype(float)

    # Drop target + experiment_id (we do NOT want ID as a feature)
    feature_df = df.drop(columns=[target_col, "experiment_id"], errors="ignore")

    # Determine window length from suffixes "_t0", "_t1", ...
    timestep_suffixes = sorted({col.split("_t")[-1] for col in feature_df.columns})
    window_len = len(timestep_suffixes)

    # Number of features
    total_feature_cols = feature_df.shape[1]
    num_features = total_feature_cols // window_len

    # Reshape → (samples, timesteps, features)
    X = feature_df.values.reshape(len(df), window_len, num_features)

    return X, y, window_len, num_features



# ---------------------------------------------------------
# BUILD GRU MODEL
# ---------------------------------------------------------
def build_gru_model(window_len, num_features, units=64, dropout_rate=0.2):

    model = Sequential([
        Input(shape=(window_len, num_features)),

        GRU(units, return_sequences=True),
        Dropout(dropout_rate),

        GRU(units // 2),
        Dropout(dropout_rate),

        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")    # binary probability output
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2),
        metrics=["accuracy"]
    )

    return model



# ---------------------------------------------------------
# TRAIN GRU WITH CLASS WEIGHTS
# ---------------------------------------------------------
def train_gru(model, X_train, y_train, X_val, y_val, epochs=40, batch_size=32):

    # Compute class weights (useful for target imbalance)
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    weights = dict(enumerate(weights))
    print("Class weights:", weights)

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
# PREDICT PROBABILITIES + HARD LABELS
# ---------------------------------------------------------
def predict_success(model, X_test, threshold=0.30):
    probs = model.predict(X_test)
    preds = (probs > threshold).astype(int)
    return preds, probs

