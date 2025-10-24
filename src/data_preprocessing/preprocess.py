import os
import cv2
import numpy as np
from mtcnn import MTCNN
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import albumentations as A

def create_directories():
    """Create necessary directories for processed data"""
    os.makedirs('data/processed/train/real', exist_ok=True)
    os.makedirs('data/processed/train/fake', exist_ok=True)
    os.makedirs('data/processed/test/real', exist_ok=True)
    os.makedirs('data/processed/test/fake', exist_ok=True)

def extract_frames(video_path, num_frames=15):
    """Extract frames from video file"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    if total_frames <= num_frames:
        # If video has fewer frames than requested, take all frames
        frame_indices = range(total_frames)
    else:
        # Otherwise, sample frames evenly
        frame_indices = sorted(random.sample(range(total_frames), num_frames))
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    return frames

def detect_face(image, detector):
    """Detect and crop face using MTCNN"""
    results = detector.detect_faces(image)
    if not results:
        return None
    
    # Get the face with the largest bounding box
    best_face = max(results, key=lambda x: x['box'][2] * x['box'][3])
    x, y, w, h = best_face['box']
    
    # Add some padding
    padding = int(max(w, h) * 0.1)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(image.shape[1] - x, w + 2 * padding)
    h = min(image.shape[0] - y, h + 2 * padding)
    
    face = image[y:y+h, x:x+w]
    return face

def augment_image(image):
    """Apply data augmentation to the image"""
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.2),
        A.HueSaturationValue(p=0.3)
    ])
    return transform(image=image)['image']

def process_video(video_path, output_dir, detector, is_real, num_frames=15, augment=True):
    """Process a single video file"""
    frames = extract_frames(video_path, num_frames)
    
    for i, frame in enumerate(frames):
        face = detect_face(frame, detector)
        if face is None:
            continue
            
        # Resize face to 224x224 (standard size for many models)
        face = cv2.resize(face, (224, 224))
        
        # Save original
        filename = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{filename}_frame{i}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        
        # Save augmented version if needed
        if augment:
            augmented = augment_image(face)
            aug_path = os.path.join(output_dir, f"{filename}_frame{i}_aug.jpg")
            cv2.imwrite(aug_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))

def main():
    # Initialize face detector
    detector = MTCNN()
    
    # Create directories
    create_directories()
    
    # Process real videos
    real_videos = [f for f in os.listdir('data/raw/real') if f.endswith(('.mp4', '.avi', '.mov'))]
    for video in tqdm(real_videos, desc="Processing real videos"):
        video_path = os.path.join('data/raw/real', video)
        process_video(video_path, 'data/processed/train/real', detector, is_real=True)
    
    # Process fake videos
    fake_videos = [f for f in os.listdir('data/raw/fake') if f.endswith(('.mp4', '.avi', '.mov'))]
    for video in tqdm(fake_videos, desc="Processing fake videos"):
        video_path = os.path.join('data/raw/fake', video)
        process_video(video_path, 'data/processed/train/fake', detector, is_real=False)
    
    # Create train/test split (move 20% of data to test set)
    for label in ['real', 'fake']:
        files = os.listdir(f'data/processed/train/{label}')
        train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
        
        # Move files to test directory
        for file in test_files:
            src = os.path.join('data/processed/train', label, file)
            dst = os.path.join('data/processed/test', label, file)
            os.rename(src, dst)

if __name__ == "__main__":
    main()
