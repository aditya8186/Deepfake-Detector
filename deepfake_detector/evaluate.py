import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.applications.efficientnet import preprocess_input


def load_manifest(manifest_path: Path):
    paths, labels = [], []
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            paths.append(row["path"]) 
            labels.append(int(row["label"]))
    return np.array(paths), np.array(labels, dtype=np.int64)


def main():
    project_root = Path(__file__).resolve().parent
    outputs_root = project_root / "outputs"
    models_root = project_root / "models"
    outputs_root.mkdir(parents=True, exist_ok=True)

    manifest_path = outputs_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("Manifest not found. Run preprocessing first.")

    # load latest available model
    model_path = None
    for name in ["model_efficient.keras", "model_efficient_final.keras", "model.keras", "model_final.keras"]:
        p = models_root / name
        if p.exists():
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError("No trained model found in models/.")

    # Provide custom_objects for Lambda(preprocess_input) and disable safe mode
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"preprocess_input": preprocess_input},
        safe_mode=False,
        compile=True,
    )

    paths, labels = load_manifest(manifest_path)
    # use the entire manifest for evaluation (or split if desired)
    xs = [np.load(p).astype("float32") / 255.0 for p in paths]
    X = np.array(xs)
    y_true = labels

    y_prob = model.predict(X, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_true, y_pred, target_names=["real", "fake"], digits=4)
    cm = confusion_matrix(y_true, y_pred)

    with open(outputs_root / "classification_report.txt", "w") as f:
        f.write(report)

    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["real", "fake"], yticklabels=["real", "fake"])
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(outputs_root / "confusion_matrix.png")
    print("Saved classification_report.txt and confusion_matrix.png in outputs/.")


if __name__ == "__main__":
    main()


