import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report, confusion_matrix

from src.data_preprocessing.data_generator import create_data_generators


def evaluate(data_dir: str, batch_size: int, sequence_length: int, frame_size: int):
    # Build generators (use explicit test dir if exists; otherwise auto-split fallback will split train)
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # For evaluation, we want the val_gen/test_gen. If test_dir exists, use it; otherwise, we will evaluate on the val split created from train.
    has_test = os.path.isdir(test_dir)

    gens = create_data_generators(
        train_dir=train_dir,
        val_dir=test_dir if has_test else train_dir,
        test_dir=None,
        batch_size=batch_size,
        sequence_length=sequence_length,
        target_size=(frame_size, frame_size),
        val_split=0.2,
        segments_per_video=1,
    )

    train_gen, val_gen = gens
    eval_gen = val_gen  # Evaluate on validation (or test if provided)

    # Load best or final model if exists
    model_path = None
    models_dir = 'models'
    candidates = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
    # Prefer best model
    bests = sorted([f for f in candidates if 'model_best_' in f], reverse=True)
    finals = sorted([f for f in candidates if 'model_final_' in f], reverse=True)
    if bests:
        model_path = os.path.join(models_dir, bests[0])
    elif finals:
        model_path = os.path.join(models_dir, finals[0])
    else:
        raise FileNotFoundError('No .keras model found in models/. Train the model first.')

    print(f'Loading model: {model_path}')
    model = tf.keras.models.load_model(model_path)

    # Gather predictions
    y_true = []
    y_prob = []

    for i in range(len(eval_gen)):
        X_batch, y_batch = eval_gen[i]
        if len(X_batch) == 0:
            continue
        preds = model.predict(X_batch, verbose=0)
        y_true.extend(y_batch.tolist())
        y_prob.extend(preds.tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    try:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    except Exception:
        auc = float('nan')

    print('\nEvaluation Metrics:')
    print(f'Accuracy : {acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall   : {recall:.4f}')
    print(f'F1-score : {f1:.4f}')
    print(f'AUC      : {auc:.4f}')

    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=['Real','Fake']))

    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate deepfake model')
    parser.add_argument('--data-dir', type=str, default='data/processed')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--sequence-length', type=int, default=10)
    parser.add_argument('--frame-size', type=int, default=128)
    args = parser.parse_args()

    evaluate(args.data_dir, args.batch_size, args.sequence_length, args.frame_size)
