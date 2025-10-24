import os
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models


def load_manifest(manifest_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    paths = []
    labels = []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["path"]) 
            labels.append(int(row["label"]))
    return np.array(paths), np.array(labels, dtype=np.int64)


def load_sequences(paths: np.ndarray) -> np.ndarray:
    seqs = []
    for p in paths:
        arr = np.load(p)
        # normalize to [0,1]
        seqs.append(arr.astype("float32") / 255.0)
    return np.array(seqs)


def build_model(frame_shape=(160, 160, 3), timesteps=32) -> tf.keras.Model:
    # CNN encoder (shared across time) using TimeDistributed
    cnn = models.Sequential([
        layers.Conv2D(32, 3, activation="relu", padding="same", input_shape=frame_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
    ], name="cnn_encoder")

    inputs = layers.Input(shape=(timesteps, *frame_shape))
    x = layers.TimeDistributed(cnn)(inputs)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    project_root = Path(__file__).resolve().parent
    outputs_root = project_root / "outputs"
    models_root = project_root / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    manifest_path = outputs_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run utils/preprocess_dataset.py first.")

    paths, labels = load_manifest(manifest_path)
    X_train_p, X_val_p, y_train, y_val = train_test_split(paths, labels, test_size=0.2, stratify=labels, random_state=42)

    X_train = load_sequences(X_train_p)
    X_val = load_sequences(X_val_p)

    timesteps, h, w, c = X_train.shape[1:]
    model = build_model(frame_shape=(h, w, c), timesteps=timesteps)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(models_root / "model.keras"), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=int(os.environ.get("EPOCHS", 15)),
        batch_size=int(os.environ.get("BATCH_SIZE", 4)),
        callbacks=callbacks,
        verbose=1,
    )

    model.save(models_root / "model_final.keras")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    with open(outputs_root / "metrics.txt", "w") as f:
        f.write(f"val_loss={val_loss}\nval_accuracy={val_acc}\n")
    print(f"Validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()


