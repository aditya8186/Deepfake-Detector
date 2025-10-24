import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DeepfakeDetector:
    def __init__(self, input_shape=(15, 224, 224, 3), num_classes=2):
        """
        Initialize the DeepfakeDetector with CNN-LSTM architecture
        
        Args:
            input_shape: Tuple of (sequence_length, height, width, channels)
            num_classes: Number of output classes (2 for binary classification)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        
    def _cnn_block(self, x, filters, kernel_size, pool_size, dropout_rate=0.3):
        """Create a CNN block with Conv2D, BatchNorm, ReLU, MaxPooling, and Dropout"""
        x = layers.TimeDistributed(
            layers.Conv2D(filters, kernel_size, padding='same', activation='relu')
        )(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D(pool_size=pool_size))(x)
        x = layers.TimeDistributed(layers.Dropout(dropout_rate))(x)
        return x
    
    def _build_model(self):
        """Build a CNN-LSTM model using MobileNetV2 backbone per frame"""
        # Expected input: (T, H, W, C)
        input_layer = layers.Input(shape=self.input_shape)
        seq_len, height, width, channels = self.input_shape

        # Build MobileNetV2 backbone (imagenet weights) for per-frame features
        base = tf.keras.applications.MobileNetV2(
            input_shape=(height, width, channels),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        base.trainable = False  # freeze initially; can unfreeze later for fine-tuning

        # TimeDistributed backbone
        x = layers.TimeDistributed(base)(input_layer)
        # Optional normalization to float32 if using mixed precision
        x = layers.TimeDistributed(layers.LayerNormalization())(x)

        # Temporal modeling
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        output = layers.Dense(self.num_classes, activation='softmax', dtype='float32')(x)

        model = models.Model(inputs=input_layer, outputs=output)
        model.summary()
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with memory optimizations"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f'Mixed precision enabled. Compute dtype: {policy.compute_dtype}, Variable dtype: {policy.variable_dtype}')
                # Enable memory growth
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception:
                        pass
                print("Enabled memory growth for GPU")
            else:
                # Ensure float32 on CPU to avoid depthwise conv issues with float16
                policy = tf.keras.mixed_precision.Policy('float32')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("No GPU detected. Using float32 precision.")
        except Exception as e:
            print(f'Precision policy setup warning: {e}')
        
        # Configure optimizer with gradient clipping and weight decay
        optimizer = optimizers.Adam(
            learning_rate=learning_rate,
            global_clipnorm=1.0,
            epsilon=1e-7
        )
        
        # Use standard cross-entropy loss
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
    
    def train(
        self, 
        train_generator, 
        val_generator, 
        epochs=50, 
        batch_size=32,
        model_save_path='models/deepfake_detector.h5',
        early_stopping_patience=5,
        reduce_lr_patience=3
    ):
        """Train the model with callbacks for early stopping and learning rate reduction"""
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=reduce_lr_patience,
                min_lr=1e-6
            ),
            callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_generator):
        """Evaluate the model on test data"""
        return self.model.evaluate(test_generator, verbose=1)
    
    def predict(self, X):
        """Make predictions on new data"""
        return self.model.predict(X, verbose=0)
    
    def save_model(self, filepath):
        """Save the entire model to a file"""
        self.model.save(filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """Load a saved model"""
        model = models.load_model(filepath)
        detector = cls()
        detector.model = model
        return detector
    
    def plot_training_history(self, history):
        """Plot training and validation metrics"""
        metrics = ['loss', 'accuracy', 'auc']
        
        plt.figure(figsize=(15, 5))
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, i)
            plt.plot(history.history[metric], label=f'Training {metric}')
            if f'val_{metric}' in history.history:
                plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'Model {metric.capitalize()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.capitalize())
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=['Real', 'Fake']):
        """Plot confusion matrix"""
        y_pred_classes = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_true, y_pred_classes)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred_classes, target_names=class_names))
