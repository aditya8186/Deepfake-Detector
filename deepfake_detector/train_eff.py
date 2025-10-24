import os
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input


def load_manifest(manifest_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    paths, labels = [], []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["path"]) 
            labels.append(int(row["label"]))
    return np.array(paths), np.array(labels, dtype=np.int64)


def make_dataset(paths: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> tf.data.Dataset:
    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))
    if shuffle:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)

    def _load_np(path_str, label):
        def _py_load(p):
            arr = np.load(p.decode("utf-8")).astype("float32")
            return arr
        seq = tf.numpy_function(_py_load, [path_str], Tout=tf.float32)
        seq.set_shape([None, 160, 160, 3])
        # scale to [0,1]
        seq = seq / 255.0
        # ensure 3 channels if loaded as grayscale by mistake
        def _ensure_rgb(x):
            ch = tf.shape(x)[-1]
            return tf.cond(
                tf.equal(ch, 1),
                lambda: tf.repeat(x, repeats=3, axis=-1),
                lambda: x,
            )
        seq = _ensure_rgb(seq)
        return seq, tf.cast(label, tf.float32)

    ds = ds.map(_load_np, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_eff_bilstm(timesteps: int = 32) -> tf.keras.Model:
    frame_shape = (160, 160, 3)

    # Try to load ImageNet weights; if channel mismatch occurs, fall back to random init
    try:
        backbone = EfficientNetB0(include_top=False, weights="imagenet", input_shape=frame_shape, pooling="avg")
    except Exception:
        backbone = EfficientNetB0(include_top=False, weights=None, input_shape=frame_shape, pooling="avg")
    backbone.trainable = False  # start frozen for stability

    inputs = layers.Input(shape=(timesteps, *frame_shape))
    # preprocess per-frame for EfficientNet
    def _preprocess(x):
        # if single-channel slipped through, convert to RGB
        ch = tf.shape(x)[-1]
        x = tf.cond(tf.equal(ch, 1), lambda: tf.repeat(x, repeats=3, axis=-1), lambda: x)
        x = preprocess_input(x * 255.0)  # EfficientNet expects 0-255
        return x
    x = layers.TimeDistributed(layers.Lambda(_preprocess))(inputs)
    x = layers.TimeDistributed(backbone)(x)
    x = layers.Bidirectional(layers.LSTM(192, return_sequences=False))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
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

    batch_size = int(os.environ.get("BATCH_SIZE", 4))
    epochs = int(os.environ.get("EPOCHS", 30))

    # infer timesteps by loading one sample header
    sample_seq = np.load(X_train_p[0])
    timesteps = sample_seq.shape[0]

    train_ds = make_dataset(X_train_p, y_train, batch_size=batch_size, shuffle=True)
    val_ds = make_dataset(X_val_p, y_val, batch_size=batch_size, shuffle=False)

    model = build_eff_bilstm(timesteps=timesteps)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(models_root / "model_efficient.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(outputs_root / "history.csv"), append=False),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Unfreeze backbone for fine-tuning last stages
    model.get_layer(index=2).layer.trainable = True  # TimeDistributed(backbone).layer is the backbone
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max(10, epochs // 3),
        callbacks=callbacks,
        verbose=1,
    )

    # Final save
    model.save(models_root / "model_efficient_final.keras")

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    with open(outputs_root / "metrics.txt", "a") as f:
        f.write(f"\n[EfficientNet+BILSTM] val_loss={val_loss}\nval_accuracy={val_acc}\n")
    print(f"Validation accuracy (EfficientNet+BILSTM): {val_acc:.4f}")


if __name__ == "__main__":
    main()


