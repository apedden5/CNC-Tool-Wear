# transformer.py  –  Keras Transformer with keras-tuner HPO

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid OpenMP clash on Windows

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

# ------------------------ Global config ------------------------

TARGET_COL   = "successful_part"
WINDOW_SIZE  = 10
NUM_CLASSES  = 2

BATCH_SIZE   = 64
EPOCHS_TUNE  = 40      # per trial
EPOCHS_FINAL = 60      # final best model

# ------------------------ Paths ------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

BASE_DIR = _project_root()
WIN_DIR  = BASE_DIR / "data" / "data_windowed_csv"

TRAIN_CSV = WIN_DIR / f"train_windows_w{WINDOW_SIZE}.csv"
VAL_CSV   = WIN_DIR / f"val_windows_w{WINDOW_SIZE}.csv"
TEST_CSV  = WIN_DIR / f"test_windows_w{WINDOW_SIZE}.csv"


# ------------------------ Data loading ------------------------

def load_windowed_csv(csv_path: Path):
    """
    Load a windowed CSV and return:
      X: (N, W, F) float32
      y: (N,) float32 (0 / 1)
    """
    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} not in {csv_path}")

    non_feat = {TARGET_COL, "experiment_id"}
    flat_cols = [c for c in df.columns if c not in non_feat]

    base_feats   = sorted({c.rsplit("_t", 1)[0] for c in flat_cols})
    time_indices = sorted({int(c.rsplit("_t", 1)[1]) for c in flat_cols})

    if len(time_indices) != WINDOW_SIZE:
        raise ValueError(f"Expected {WINDOW_SIZE} timesteps, found {len(time_indices)}")

    X_list = []
    for t in time_indices:
        cols_t = [f"{feat}_t{t}" for feat in base_feats]
        X_t = df[cols_t].values.astype("float32")
        X_list.append(X_t)

    X = np.stack(X_list, axis=1)  # (N, W, F)
    y = df[TARGET_COL].values.astype("float32")

    return X, y


# ------------------------ Transformer block ------------------------

# time_steps / num_features are needed inside the block just like in the lab
X_train_tmp, _ = load_windowed_csv(TRAIN_CSV)
time_steps   = X_train_tmp.shape[1]
num_features = X_train_tmp.shape[2]
num_classes  = NUM_CLASSES

print("time_steps :", time_steps)
print("num_features:", num_features)
print("num_classes :", num_classes)


class TransformerEncoderBlock(Layer):
    def __init__(self, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        # Multi-head self-attention
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=1)

        # Feed-forward via Conv1D(1x1)
        self.conv1 = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")
        self.conv2 = Conv1D(filters=num_features, kernel_size=1)

        # LayerNorm + Dropout
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1   = Dropout(rate)
        self.dropout2   = Dropout(rate)

    def call(self, inputs, training=False):
        # --- First residual: attention ---
        attn_output = self.att(inputs, inputs)                 # self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)           # residual + norm

        # --- Second residual: feed-forward via Conv1D(1x1) ---
        ffn_output = self.conv1(out1)
        ffn_output = self.conv2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)              # residual + norm


# ------------------------ keras-tuner model builder ------------------------

def build_transformer_model(hp: kt.HyperParameters) -> Model:
    num_heads = hp.Choice("num_heads", [2, 4])
    ff_dim    = hp.Choice("ff_dim", [64, 128])
    num_blocks = hp.Choice("num_blocks", [1, 2, 3])
    dense_units = hp.Int("dense_units", min_value=32, max_value=128, step=32)
    dropout_rate = hp.Choice("dropout_rate", [0.1, 0.2, 0.3])
    lr = hp.Choice("lr", [1e-3, 5e-4, 2e-4])

    inputs = Input(shape=(time_steps, num_features))
    x = inputs

    for _ in range(num_blocks):
        x = TransformerEncoderBlock(
            num_heads=num_heads,
            ff_dim=ff_dim,
            rate=dropout_rate,
        )(x)

    x = Flatten()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1, activation="sigmoid")(x)  # binary classification

    model = Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ------------------------ Main script ------------------------

def main():
    # ---- Load data ----
    X_train, y_train = load_windowed_csv(TRAIN_CSV)
    X_val,   y_val   = load_windowed_csv(VAL_CSV)
    X_test,  y_test  = load_windowed_csv(TEST_CSV)

    print("X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)

    # ---- keras-tuner GridSearch ----
    keras.backend.clear_session()

    tuner = kt.GridSearch(
        hypermodel=build_transformer_model,
        objective="val_accuracy",
        max_trials=25,                # small but useful search space
        directory="transformer_hpo",
        project_name="transformer_search",
        overwrite=True,
    )

    tuner.search(
        X_train, y_train,
        epochs=EPOCHS_TUNE,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=1,
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("\nBest hyperparameters:")
    for name in ["num_heads", "ff_dim", "num_blocks",
                 "dense_units", "dropout_rate", "lr"]:
        print(f"  {name}: {best_hp.get(name)}")

    # ---- Build and train final model with best HPs ----
    best_model = tuner.hypermodel.build(best_hp)

    es = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    )

    history = best_model.fit(
        X_train, y_train,
        epochs=EPOCHS_FINAL,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[es],
        verbose=1,
    )

    # ---- Evaluate on test set ----
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)

    y_prob = best_model.predict(X_test, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype(int)

    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\nTest loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Confusion matrix:")
    print(cm)
    print(f"ROC AUC: {roc_auc:.4f}")

    # ---- Plots ----
    epochs = range(1, len(history.history["loss"]) + 1)

    # Loss curve
    plt.figure()
    plt.plot(epochs, history.history["loss"], label="Train Loss")
    plt.plot(epochs, history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Transformer – Successful Part (Train vs Val Loss)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "transformer_keras_loss.png", dpi=200)
    plt.show()

    # Confusion matrix
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)")
    plt.colorbar(im)
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, [0, 1])
    plt.yticks(tick_marks, [0, 1])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    plt.savefig(BASE_DIR / "transformer_keras_confusion.png", dpi=200)
    plt.show()

    # ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve – Successful Part (Test Set)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "transformer_keras_roc.png", dpi=200)
    plt.show()

    # Save final model
    best_model.save(BASE_DIR / "transformer_keras_best.h5")
    print(f"Saved final Keras model to {BASE_DIR / 'transformer_keras_best.h5'}")


if __name__ == "__main__":
    main()
