import os
import argparse
import cv2
import numpy as np
import tensorflow as tf

DEFAULT_SEQ_LEN = 10
DEFAULT_FRAME_SIZE = 128


def load_latest_model(models_dir: str = 'models') -> str:
    """Return path to the latest .keras model, preferring best over final."""
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    candidates = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
    if not candidates:
        raise FileNotFoundError("No .keras model found in models/. Train the model first.")
    bests = sorted([f for f in candidates if 'model_best_' in f], reverse=True)
    finals = sorted([f for f in candidates if 'model_final_' in f], reverse=True)
    chosen = bests[0] if bests else finals[0]
    return os.path.join(models_dir, chosen)


def extract_frames(video_path: str, target_size=(DEFAULT_FRAME_SIZE, DEFAULT_FRAME_SIZE)):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size)
        frame = frame.astype('float32') / 255.0
        frames.append(frame)
    cap.release()
    return np.array(frames)  # (N, H, W, 3)


def sample_segments(frames: np.ndarray, sequence_length: int, segments: int = 5):
    """Sample multiple segments of length T from the frames. If frames < T, pad by repeating last frame."""
    n = len(frames)
    if n == 0:
        raise ValueError("No frames extracted from video.")
    segs = []
    for _ in range(max(1, segments)):
        if n <= sequence_length:
            # pad last frame
            pad = [frames[-1]] * (sequence_length - n)
            seq = np.concatenate([frames, np.array(pad)], axis=0) if len(pad) > 0 else frames
        else:
            start = np.random.randint(0, n - sequence_length + 1)
            seq = frames[start:start + sequence_length]
        segs.append(seq)
    return np.array(segs)  # (S, T, H, W, 3)


def predict_video(video_path: str, sequence_length: int = DEFAULT_SEQ_LEN, frame_size: int = DEFAULT_FRAME_SIZE,
                  segments: int = 5, models_dir: str = 'models'):
    model_path = load_latest_model(models_dir)
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    frames = extract_frames(video_path, (frame_size, frame_size))
    segs = sample_segments(frames, sequence_length=sequence_length, segments=segments)
    # Predict per segment, then average
    probs = model.predict(segs, verbose=0)  # (S, num_classes)
    mean_prob = probs.mean(axis=0)
    pred_class = int(np.argmax(mean_prob))
    classes = ['Real', 'Fake']
    return {
        'model_path': model_path,
        'prediction': classes[pred_class],
        'confidence': float(mean_prob[pred_class]),
        'probs': mean_prob.tolist(),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict deepfake on a video file')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--sequence-length', type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument('--frame-size', type=int, default=DEFAULT_FRAME_SIZE)
    parser.add_argument('--segments', type=int, default=5)
    parser.add_argument('--models-dir', type=str, default='models')
    args = parser.parse_args()

    result = predict_video(args.video, args.sequence_length, args.frame_size, args.segments, args.models_dir)
    print(f"Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
    print(f"Probs: {result['probs']}")
