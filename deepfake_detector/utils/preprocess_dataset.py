import os
import sys
import csv
import uuid
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

# Allow running directly: add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.frame_extractor import extract_frames
from utils.face_detection import crop_primary_face


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_video(video_path: Path, out_dir: Path, max_frames: int = 32,
                  face_size=(160, 160)) -> Path:
    frames = extract_frames(str(video_path), max_frames=max_frames, resize=(224, 224))
    if not frames:
        return Path()

    crops: List[np.ndarray] = []
    for frame in frames:
        crop = crop_primary_face(frame, target_size=face_size)
        crops.append(crop)

    seq = np.stack(crops).astype("uint8")  # shape: (T, H, W, 3)
    seq_id = f"{video_path.stem}__{uuid.uuid4().hex[:8]}"
    out_path = out_dir / f"{seq_id}.npy"
    np.save(out_path, seq)
    return out_path


def build_manifest(data_root: Path, outputs_root: Path, max_frames: int = 32) -> Path:
    sequences_dir = outputs_root / "sequences"
    ensure_dir(sequences_dir)
    manifest_path = outputs_root / "manifest.csv"

    rows = []
    # label mapping: real=0, fake=1
    for label_name, label in [("real", 0), ("fake", 1)]:
        src_dir = data_root / label_name
        if not src_dir.exists():
            continue
        videos = list(src_dir.glob("*.mp4")) + list(src_dir.glob("*.avi"))
        for v in tqdm(videos, desc=f"Processing {label_name}"):
            try:
                out = process_video(v, sequences_dir, max_frames=max_frames)
            except Exception:
                out = Path()
            if out and out.exists():
                rows.append([str(out), label])

    # write manifest
    ensure_dir(outputs_root)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])  # header
        writer.writerows(rows)

    return manifest_path


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_root = project_root / "data"
    outputs_root = project_root / "outputs"
    ensure_dir(outputs_root)

    max_frames = int(os.environ.get("MAX_FRAMES", 32))
    manifest = build_manifest(data_root, outputs_root, max_frames=max_frames)
    print(f"Wrote manifest: {manifest}")


if __name__ == "__main__":
    main()

