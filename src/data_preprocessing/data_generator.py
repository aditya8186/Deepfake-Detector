import os
import re
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
from collections import defaultdict
import random

class FrameSequenceGenerator(Sequence):
    def __init__(self, data_dir, batch_size=8, sequence_length=15, target_size=(224, 224), 
                 augment=False, num_classes=2, shuffle=True, allowed_keys=None, segments_per_video=1):
        """
        Initialize the frame sequence generator
        
        Args:
            data_dir: Directory containing 'real' and 'fake' subdirectories with frame images
            batch_size: Number of sequences per batch
            sequence_length: Number of frames per sequence
            target_size: Target size for resizing frames
            augment: Whether to apply data augmentation
            num_classes: Number of output classes
            shuffle: Whether to shuffle the data
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.target_size = target_size
        self.augment = augment
        self.num_classes = num_classes
        self.shuffle = shuffle
        # Optional whitelist of (class_name, video_id) pairs to include
        # If provided, only these videos will be considered
        self.allowed_keys = set(allowed_keys) if allowed_keys is not None else None
        # How many segments to sample per video per epoch
        self.segments_per_video = max(1, int(segments_per_video))
        
        # Class indices mapping
        self.class_indices = {'real': 0, 'fake': 1}
        
        # Scan directory and group frames by video ID
        self.video_frames = defaultdict(list)
        self.labels = []
        
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            # Get all frame files in the class directory
            for frame_file in os.listdir(class_dir):
                if frame_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Extract video ID from filename using regex.
                    # Expected formats like:
                    #  - 01__exit_phone_room_frame0.jpg
                    #  - 01_02__exit_phone_room__YVGY8LOK_frame123_aug.jpg
                    # We capture everything before '_frame<digits>' ignoring optional '_aug'.
                    m = re.match(r"^(.*)_frame\d+(?:_aug)?\.(?:jpg|jpeg|png)$", frame_file, flags=re.IGNORECASE)
                    if m:
                        video_id = m.group(1)
                    else:
                        # Fallback: remove trailing extension and try rsplit on '_frame'
                        base = os.path.splitext(frame_file)[0]
                        base = base.replace('_aug', '')
                        video_id = base.rsplit('_frame', 1)[0]
                    frame_path = os.path.join(class_dir, frame_file)
                    self.video_frames[(class_name, video_id)].append(frame_path)
        
        # Create list of (video_id, label) pairs and keep key mapping
        self.video_list = []  # list of tuples (video_id, label)
        self.keys = []        # aligned list of (class_name, video_id)
        for (class_name, video_id), frames in self.video_frames.items():
            # Apply whitelist filter if provided
            if self.allowed_keys is not None and (class_name, video_id) not in self.allowed_keys:
                continue
            if len(frames) >= sequence_length:  # Only include videos with enough frames
                self.video_list.append((video_id, self.class_indices[class_name.lower()]))
                self.keys.append((class_name, video_id))
                frames.sort()  # Ensure frames are in order
        
        # Total logical samples: videos * segments_per_video
        self.num_videos = len(self.video_list)
        self.num_samples = self.num_videos * self.segments_per_video
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = []
        y_batch = []
        
        for idx in batch_indices:
            # Map global idx to base video idx and segment idx
            base_video_idx = idx % self.num_videos
            video_id, label = self.video_list[base_video_idx]
            class_name, _ = self.keys[base_video_idx]
            frames = self._load_video_frames_for_key(class_name, video_id)
            if frames is not None:
                # Sample a random window of sequence_length within the frames
                if len(frames) > self.sequence_length:
                    start_idx = random.randint(0, len(frames) - self.sequence_length)
                    sel_frames = frames[start_idx:start_idx + self.sequence_length]
                else:
                    sel_frames = frames
                # Load pixel data for selected frames
                seq = self._load_and_preprocess_sequence(sel_frames)
                if seq is None:
                    continue
                X_batch.append(seq)
                y_batch.append(label)
        
        # Convert to numpy arrays if we have any
        if len(X_batch) == 0:
            # Return minimal shaped arrays to avoid errors
            X_batch = np.zeros((0, self.sequence_length, self.target_size[1], self.target_size[0], 3), dtype='float32')
            y_batch = np.zeros((0,), dtype='int32')
        else:
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
        
        # No padding; allow smaller final batch
        
        return X_batch, y_batch
    
    def _load_video_frames_for_key(self, class_name, video_id):
        # Return sorted list of frame file paths for this (class_name, video_id)
        frames = self.video_frames.get((class_name, video_id), [])
        frames.sort()
        return frames

    def _load_and_preprocess_sequence(self, frame_paths):
        sequence = []
        for frame_path in frame_paths:
            img = cv2.imread(frame_path)
            if img is None:
                continue
            # Convert BGR to RGB and resize
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size)
            # Normalize to [0, 1]
            img = img.astype('float32') / 255.0
            # Apply augmentation if needed
            if self.augment and random.random() > 0.5:
                img = self._augment_frame(img)
            sequence.append(img)
        if len(sequence) < self.sequence_length:
            return None
        sequence = np.array(sequence)
        if len(sequence.shape) == 3:
            sequence = np.expand_dims(sequence, axis=-1)
        return sequence
    
    def _augment_frame(self, img):
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img)
        
        # Random brightness adjustment
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            img = np.clip(img * brightness, 0, 1)
            
        return img
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_data_generators(train_dir, val_dir, test_dir=None, batch_size=8, 
                         sequence_length=15, target_size=(224, 224), val_split=0.2, use_train_for_val_if_empty=True, segments_per_video=1):
    """
    Create data generators for training and validation (and optionally testing)
    """
    # Determine if validation directory has any image files
    def _dir_has_images(path):
        try:
            if not os.path.isdir(path):
                return False
            for cls in ("real", "fake"):
                cls_path = os.path.join(path, cls)
                if os.path.isdir(cls_path):
                    for fname in os.listdir(cls_path):
                        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                            return True
            return False
        except Exception:
            return False

    val_has_images = _dir_has_images(val_dir)

    if val_has_images:
        # Standard behavior: separate train and val directories
        train_gen = FrameSequenceGenerator(
            train_dir,
            batch_size=batch_size,
            sequence_length=sequence_length,
            target_size=target_size,
            augment=True,
            shuffle=True,
            segments_per_video=segments_per_video
        )
        val_gen = FrameSequenceGenerator(
            val_dir,
            batch_size=batch_size,
            sequence_length=sequence_length,
            target_size=target_size,
            augment=False,
            shuffle=False,
            segments_per_video=segments_per_video
        )
    else:
        # Fallback: split the train directory into train/val by video IDs
        # First, scan all videos in train_dir
        temp_gen = FrameSequenceGenerator(
            train_dir,
            batch_size=batch_size,
            sequence_length=sequence_length,
            target_size=target_size,
            augment=True,
            shuffle=True,
            segments_per_video=segments_per_video
        )
        # Build list of unique video keys (class_name, video_id)
        keys = list(temp_gen.video_frames.keys())
        import random as _random
        _random.shuffle(keys)
        split_idx = int(len(keys) * (1.0 - val_split))
        train_keys = set(keys[:split_idx])
        val_keys = set(keys[split_idx:])

        train_gen = FrameSequenceGenerator(
            train_dir,
            batch_size=batch_size,
            sequence_length=sequence_length,
            target_size=target_size,
            augment=True,
            shuffle=True,
            allowed_keys=train_keys,
            segments_per_video=segments_per_video
        )
        val_gen = FrameSequenceGenerator(
            train_dir,
            batch_size=batch_size,
            sequence_length=sequence_length,
            target_size=target_size,
            augment=False,
            shuffle=False,
            allowed_keys=val_keys,
            segments_per_video=segments_per_video
        )
    
    if test_dir:
        test_gen = FrameSequenceGenerator(
            test_dir,
            batch_size=batch_size,
            sequence_length=sequence_length,
            target_size=target_size,
            augment=False,
            shuffle=False,
            segments_per_video=segments_per_video
        )
        return train_gen, val_gen, test_gen
    
    return train_gen, val_gen
