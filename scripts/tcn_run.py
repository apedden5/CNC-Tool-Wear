import pandas as pd
import numpy as np
import tensorflow as tf
from tcn import TCN #Temporal Convolutional Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# LOAD WINDOWED CSV (generic, detects window_len + features)
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


# BUILD TCN MODEL
def build_tcn_model(window_len, num_features, nb_filters=64, kernel_size=3, dropout_rate=0.2):

    model = Sequential([
        Input(shape=(window_len, num_features)),

        TCN(nb_filters=nb_filters, kernel_size=kernel_size, dropout_rate=dropout_rate, return_sequences=False),

        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")    # Output single vlaue = probability of worn tool
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

def train_tcn_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )
    return history

def predict_tcn_model_wear(model, X_test):
    predictions = model.predict(X_test)
    return predictions