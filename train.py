import os
import sys
# Ensure project root is on sys.path so 'src' imports work when running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
import json
import time
import argparse
import psutil
import random
import numpy as np
import tensorflow as tf
from datetime import datetime
from src.models.cnn_lstm import DeepfakeDetector
from src.data_preprocessing.data_generator import create_data_generators

def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Current memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"Available memory: {psutil.virtual_memory().available / 1024 / 1024:.2f} MB")
    
    # Print GPU memory info if available
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                mem_info = tf.config.experimental.get_memory_info('GPU:0')
                print(f"GPU Memory - Current: {mem_info['current'] / 1024**2:.2f}MB, "
                      f"Peak: {mem_info['peak'] / 1024**2:.2f}MB")
        except Exception as e:
            print(f"Could not get GPU memory info: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the processed data directory')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training (reduced to prevent OOM)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs to train')
    parser.add_argument('--sequence-length', type=int, default=10, help='Number of frames per sequence (reduced to save memory)')
    parser.add_argument('--frame-size', type=int, default=160, help='Frame size (width and height, reduced to save memory)')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--logs-dir', type=str, default='logs', help='Directory to save logs')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split ratio if val dir is empty')
    parser.add_argument('--segments-per-video', type=int, default=4, help='How many random segments to sample per video per epoch')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Number of warmup epochs for LR schedule')
    parser.add_argument('--unfreeze-epoch', type=int, default=5, help='Epoch at which to unfreeze MobileNetV2 backbone for fine-tuning (set -1 to disable)')
    parser.add_argument('--fine-tune-lr', type=float, default=1e-5, help='Max learning rate after unfreezing backbone')
    parser.add_argument('--class-weights', action='store_true', help='Enable class weighting to address imbalance')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    def set_seed(seed: int):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
    set_seed(args.seed)

    # Print memory information
    print_memory_usage()

    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = os.path.join(args.model_dir, f'model_{timestamp}.keras')
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    
    # Save training configuration
    config = {
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'sequence_length': args.sequence_length,
        'learning_rate': args.learning_rate,
        'input_shape': (args.sequence_length, 224, 224, 3),
        'num_classes': 2,
        'model_save_path': model_save_path,
        'val_split': args.val_split,
        'segments_per_video': args.segments_per_video,
        'seed': args.seed,
        'warmup_epochs': args.warmup_epochs,
        'unfreeze_epoch': args.unfreeze_epoch,
        'fine_tune_lr': args.fine_tune_lr,
        'class_weights': args.class_weights,
        'timestamp': timestamp
    }
    
    with open(os.path.join(args.logs_dir, f'config_{timestamp}.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\nTraining Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # Create data generators
    print("\nCreating data generators...")
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'test')  # Using test as validation
    
    # Print data directory info
    print(f"Training data directory: {train_dir}")
    print(f"Validation data directory: {val_dir}")
    
    # Define target size - using smaller size to save memory
    target_size = (args.frame_size, args.frame_size)
    print(f"Using frame size: {target_size}")
    
    # Enable memory growth for GPU to prevent OOM
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Enabled memory growth for GPU")
        except RuntimeError as e:
            print(e)
    
    # Create data generators with memory-efficient settings
    train_gen, val_gen = create_data_generators(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        target_size=target_size,
        val_split=args.val_split,
        segments_per_video=args.segments_per_video
    )
    
    # ----------------------
    # Dataset diagnostics
    # ----------------------
    def print_generator_diagnostics(name, gen):
        try:
            total_groups = len(gen.video_frames)
            usable_sequences = len(gen.video_list)
            # Compute per-class group counts and how many meet sequence_length
            per_class_counts = {c: 0 for c in gen.class_indices}
            per_class_usable = {c: 0 for c in gen.class_indices}
            frames_per_group = []
            for (class_name, video_id), frames in gen.video_frames.items():
                per_class_counts[class_name] = per_class_counts.get(class_name, 0) + 1
                frames_per_group.append(len(frames))
            # Build set of usable video_ids for quick lookup
            usable_ids = set([vid for (vid, _) in gen.video_list])
            for (class_name, video_id), frames in gen.video_frames.items():
                if video_id in usable_ids:
                    per_class_usable[class_name] = per_class_usable.get(class_name, 0) + 1
            print(f"\n[{name}] groups (videos detected): {total_groups}")
            for cls in per_class_counts:
                print(f"[{name}] {cls}: groups={per_class_counts[cls]}, usable(seqs>={gen.sequence_length})={per_class_usable[cls]}")
            if frames_per_group:
                arr = np.array(frames_per_group)
                print(f"[{name}] frames/group: min={arr.min()}, p25={np.percentile(arr,25):.1f}, median={np.median(arr):.1f}, p75={np.percentile(arr,75):.1f}, max={arr.max()}")
            # Show a few example filenames to verify naming convention
            print(f"[{name}] example files (up to 3 per class):")
            for cls in list(gen.class_indices.keys()):
                cls_dir = os.path.join(gen.data_dir, cls)
                if os.path.isdir(cls_dir):
                    samples = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg",".jpeg",".png"))][:3]
                    print(f"  {cls}: {samples}")
        except Exception as diag_e:
            print(f"[Diagnostics] Could not compute diagnostics for {name}: {diag_e}")
    
    print_generator_diagnostics("train", train_gen)
    print_generator_diagnostics("val", val_gen)
    
    # Print memory usage info
    print_memory_usage()
    
    # Get input shape from the first batch to ensure compatibility
    sample_X, _ = train_gen[0]
    input_shape = sample_X.shape[1:]  # Remove batch dimension
    
    # Update config with actual input shape
    config['input_shape'] = input_shape
    
    # Print dataset statistics (true usable sequences based on grouping)
    val_len = len(val_gen)
    has_validation = val_len > 0
    true_train_sequences = len(train_gen.video_list) if hasattr(train_gen, 'video_list') else len(train_gen) * args.batch_size
    true_val_sequences = len(val_gen.video_list) if hasattr(val_gen, 'video_list') else (val_len * args.batch_size)
    print(f"\nTraining usable sequences: {true_train_sequences}")
    print(f"Validation usable sequences: {true_val_sequences}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.sequence_length}")
    print(f"Input shape: {input_shape}")
    if not has_validation:
        print("Warning: No validation samples found. Skipping validation and early stopping.")
    
    # Create and compile the model
    print("\nCreating and compiling model...")
    model = DeepfakeDetector(
        input_shape=input_shape,
        num_classes=config['num_classes']
    )
    model.compile_model(learning_rate=args.learning_rate)
    
    # Gradient accumulation parameters
    accumulation_steps = 4  # Accumulate gradients over 4 batches
    effective_batch_size = args.batch_size * accumulation_steps
    print(f"Using gradient accumulation with effective batch size: {effective_batch_size}")
    
    # Custom training loop for gradient accumulation
    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_gen,
        output_types=(tf.float32, tf.int32),
        output_shapes=([None] + list(input_shape), [None])
    ).prefetch(tf.data.AUTOTUNE)
    
    if has_validation:
        val_dataset = tf.data.Dataset.from_generator(
            lambda: val_gen,
            output_types=(tf.float32, tf.int32),
            output_shapes=([None] + list(input_shape), [None])
        ).prefetch(tf.data.AUTOTUNE)
    else:
        val_dataset = None
    
    # Optimizer and loss with LR schedule (cosine decay with warmup)
    steps_per_epoch = max(1, len(train_gen))
    total_steps = steps_per_epoch * max(1, args.epochs)
    warmup_steps = steps_per_epoch * max(0, args.warmup_epochs)

    class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, lr_max, warmup_steps, total_steps):
            super().__init__()
            self.lr_max = tf.cast(lr_max, tf.float32)
            self.warmup_steps = tf.cast(warmup_steps, tf.float32)
            self.total_steps = tf.cast(total_steps, tf.float32)

        def __call__(self, step):
            step = tf.cast(step, tf.float32)
            # Linear warmup phase
            def warmup():
                return self.lr_max * (step / tf.maximum(1.0, self.warmup_steps))
            # Cosine decay phase
            def cosine():
                progress = (step - self.warmup_steps) / tf.maximum(1.0, (self.total_steps - self.warmup_steps))
                progress = tf.clip_by_value(progress, 0.0, 1.0)
                return 0.5 * self.lr_max * (1.0 + tf.cos(np.pi * progress))
            return tf.cond(step < self.warmup_steps, warmup, cosine)

        def get_config(self):
            return {
                'lr_max': float(self.lr_max.numpy()),
                'warmup_steps': float(self.warmup_steps.numpy()),
                'total_steps': float(self.total_steps.numpy()),
            }

    base_lr_max = args.learning_rate
    learning_rate_fn = WarmupCosine(lr_max=base_lr_max, warmup_steps=warmup_steps, total_steps=total_steps)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn, clipnorm=1.0)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    # Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
    
    # Training step function with gradient accumulation
    @tf.function
    def train_step(images, labels, accumulation_steps):
        with tf.GradientTape() as tape:
            predictions = model.model(images, training=True)
            loss = loss_fn(labels, predictions) / tf.cast(accumulation_steps, tf.float32)
        
        gradients = tape.gradient(loss, model.model.trainable_variables)
        return loss, gradients
    
    try:
        # Training loop with gradient accumulation
        best_val_accuracy = 0.0
        patience_counter = 0
        
        print("\nStarting training...")
        # Prepare class weights if enabled
        class_weights = None
        if args.class_weights and hasattr(train_gen, 'video_list'):
            # video_list contains tuples (video_id, label)
            labels_list = [lbl for (_, lbl) in train_gen.video_list]
            if labels_list:
                import collections
                cnt = collections.Counter(labels_list)
                total = sum(cnt.values())
                class_weights = {c: total / (len(cnt) * cnt[c]) for c in cnt}
                print(f"Using class weights: {class_weights}")
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            start_time = time.time()
            
            # Reset metrics
            train_loss.reset_state()
            train_accuracy.reset_state()
            val_loss.reset_state()
            val_accuracy.reset_state()
            
            # Unfreeze backbone for fine-tuning at the chosen epoch
            if args.unfreeze_epoch >= 0 and epoch == args.unfreeze_epoch:
                try:
                    # Find MobileNetV2 inside TimeDistributed
                    for layer in model.model.layers:
                        if isinstance(layer, tf.keras.layers.TimeDistributed):
                            try:
                                inner = layer.layer
                                if inner.name.lower().startswith('mobilenetv2'):
                                    inner.trainable = True
                                    print("Unfroze MobileNetV2 backbone for fine-tuning.")
                                    break
                            except Exception:
                                continue
                    # Switch to a lower LR schedule for fine-tuning
                    fine_total_steps = steps_per_epoch * max(1, args.epochs - epoch)
                    fine_warmup = max(0, int(0.2 * fine_total_steps))
                    fine_lr = WarmupCosine(lr_max=args.fine_tune_lr, warmup_steps=fine_warmup, total_steps=fine_total_steps)
                    optimizer.learning_rate = fine_lr
                    print(f"Switched LR schedule for fine-tuning. lr_max={args.fine_tune_lr}")
                except Exception as ue:
                    print(f"Warning: could not unfreeze backbone: {ue}")

            # Training loop with gradient accumulation
            for batch, (images, labels) in enumerate(train_dataset):
                # Forward pass and gradient accumulation
                # Apply class weights if enabled
                if class_weights is not None:
                    # Compute weighted loss manually
                    with tf.GradientTape() as tape:
                        predictions = model.model(images, training=True)
                        weights = tf.gather([class_weights.get(0, 1.0), class_weights.get(1, 1.0)], tf.cast(labels, tf.int32))
                        losses = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                        loss = tf.reduce_mean(losses * tf.cast(weights, tf.float32)) / tf.cast(accumulation_steps, tf.float32)
                    grads = tape.gradient(loss, model.model.trainable_variables)
                    batch_loss = loss
                else:
                    batch_loss, grads = train_step(images, labels, accumulation_steps)
                
                # Accumulate gradients
                if (batch + 1) % accumulation_steps == 0:
                    optimizer.apply_gradients(zip(grads, model.model.trainable_variables))
                    
                    # Update metrics
                    predictions = model.model(images, training=False)
                    train_loss(batch_loss * accumulation_steps)
                    train_accuracy(labels, predictions)
                    
                    # Print progress
                    if batch % (10 * accumulation_steps) == 0:
                        print(f'Epoch {epoch + 1}, Batch {batch // accumulation_steps}, Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result() * 100:.2f}%')
            
            # Validation (if available)
            if has_validation and val_dataset is not None:
                for val_images, val_labels in val_dataset.take(10):  # Limit validation batches to save memory
                    val_predictions = model.model(val_images, training=False)
                    v_loss = loss_fn(val_labels, val_predictions)
                    val_loss(v_loss)
                    val_accuracy(val_labels, val_predictions)
            
            # Print metrics
            if has_validation and val_dataset is not None:
                print(f'Epoch {epoch + 1}, '
                      f'Loss: {train_loss.result():.4f}, '
                      f'Accuracy: {train_accuracy.result() * 100:.2f}%, '
                      f'Val Loss: {val_loss.result():.4f}, '
                      f'Val Accuracy: {val_accuracy.result() * 100:.2f}%')
            else:
                print(f'Epoch {epoch + 1}, '
                      f'Loss: {train_loss.result():.4f}, '
                      f'Accuracy: {train_accuracy.result() * 100:.2f}%, '
                      f'Val Loss: N/A, Val Accuracy: N/A (no validation)')
            
            # Memory usage info
            print_memory_usage()
            
            # Early stopping and model checkpoint (only if validation is available)
            if has_validation and val_dataset is not None:
                if val_accuracy.result() > best_val_accuracy:
                    best_val_accuracy = val_accuracy.result()
                    model_save_path = os.path.join(args.model_dir, f"model_best_{timestamp}.keras")
                    model.model.save(model_save_path)
                    print(f"Saved best model to {model_save_path}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 10:  # Early stopping patience
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
        
        # Save final model
        final_model_path = os.path.join(args.model_dir, f"model_final_{timestamp}.keras")
        model.model.save(final_model_path)
        print(f"\nTraining complete!")
        print(f"Final model saved to: {final_model_path}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

# The plot_training_history method is now part of the DeepfakeDetector class

if __name__ == '__main__':
    # Set memory growth for GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("GPU memory growth enabled")
        except RuntimeError as e:
            print(e)
    
    main()
