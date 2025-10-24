import cv2
from typing import List, Tuple


def sample_frame_indices(frame_count: int, max_frames: int) -> List[int]:
    max_frames = max(1, int(max_frames))
    if frame_count <= max_frames:
        return list(range(frame_count))
    step = frame_count / max_frames
    return [int(i * step) for i in range(max_frames)]


def extract_frames(video_path: str, max_frames: int = 32, resize: Tuple[int, int] = (224, 224)) -> List:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_frame_indices(frame_count, max_frames)

    frames = []
    target_w, target_h = resize
    cur = 0
    target_set = set(idxs)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cur in target_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            frames.append(frame)
        cur += 1
        if len(frames) >= len(idxs):
            break

    cap.release()
    return frames

