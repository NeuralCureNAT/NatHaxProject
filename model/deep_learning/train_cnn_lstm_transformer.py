"""
Deep Learning Pipeline for EEG Migraine Severity Prediction
Training CNN-LSTM and Transformer Models Simultaneously
Uses: chbmit_ictal_raw_data.csv and chbmit_preictal_raw_data.csv
"""

import numpy as np
import pandas as pd
import sys
import os
import time

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.regularizers import l1_l2
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow for compatibility
try:
    tf.config.run_functions_eagerly(True)
except:
    tf.config.experimental_run_functions_eagerly(True)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class DeepEEGProcessor:
    """Deep learning data processor for EEG data"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        
    def prepare_deep_learning_data(self, data, labels, sequence_length=64):
        """Prepare data for deep learning models"""
        from eeg_migraine_classifier import EEGDataProcessor
        
        base_processor = EEGDataProcessor()
        base_processor.severity_map = {'nonictal': 0.0, 'preictal': 0.5, 'ictal': 1.0}
        
        # Convert labels to severity scores
        y = np.array([base_processor.severity_map[label] for label in labels])
        
        # Scale data
        X_scaled = self.scaler.fit_transform(data.values)
        
        # Handle NaN values
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Create sequences
        X_sequences = self._create_sequences(X_scaled, sequence_length)
        y_sequences = y[sequence_length-1:]
        
        # Ensure all data is numpy arrays with proper dtype
        X_scaled = np.array(X_scaled, dtype=np.float32)
        X_sequences = np.array(X_sequences, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        y_sequences = np.array(y_sequences, dtype=np.float32)
        
        return X_scaled, X_sequences, y, y_sequences, sequence_length
    
    def _create_sequences(self, data, sequence_length):
        """Create sequences from time-series data"""
        if hasattr(data, 'numpy'):
            data = data.numpy()
        data = np.array(data, dtype=np.float32)
        
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i+sequence_length])
        sequences = np.array(sequences, dtype=np.float32)
        return sequences

class CNN_LSTM_Model:
    """Hybrid CNN-LSTM model combining spatial and temporal features"""
    
    def __init__(self, input_shape, num_features):
        self.input_shape = input_shape
        self.num_features = num_features
        self.model = self._build_model()
    
    def _build_model(self):
        """Build CNN-LSTM hybrid model"""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # CNN layers for spatial feature extraction
            layers.Conv1D(filters=128, kernel_size=7, activation='relu', padding='same',
                         kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same',
                         kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            # LSTM layers for temporal sequence learning
            layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3,
                       kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            layers.BatchNormalization(),
            
            layers.LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3,
                       kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            layers.BatchNormalization(),
            
            # Dense layers
            layers.Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer
            layers.Dense(1, activation='linear')
        ])
        
        return model

class Transformer_Model:
    """Transformer model with multi-head attention for EEG data"""
    
    def __init__(self, input_shape, num_features, num_heads=8, ff_dim=256):
        self.input_shape = input_shape
        self.num_features = num_features
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.model = self._build_model()
    
    def _transformer_block(self, inputs, num_heads, ff_dim, dropout_rate=0.3):
        """Transformer encoder block"""
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=ff_dim // num_heads, dropout=dropout_rate
        )(inputs, inputs)
        attention_output = layers.Dropout(dropout_rate)(attention_output)
        out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed-forward network
        ffn_output = layers.Dense(ff_dim, activation='relu')(out1)
        ffn_output = layers.Dense(inputs.shape[-1])(ffn_output)
        ffn_output = layers.Dropout(dropout_rate)(ffn_output)
        out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
        return out2
    
    def _build_model(self):
        """Build Transformer model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Positional encoding (simplified)
        x = layers.Dense(self.num_features)(inputs)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Transformer blocks
        x = self._transformer_block(x, self.num_heads, self.ff_dim, dropout_rate=0.3)
        x = self._transformer_block(x, self.num_heads, self.ff_dim, dropout_rate=0.3)
        x = self._transformer_block(x, self.num_heads, self.ff_dim, dropout_rate=0.2)
        
        # Global average pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = models.Model(inputs, outputs)
        return model

class CNN_LSTM_Transformer_Pipeline:
    """Pipeline for training CNN-LSTM and Transformer models simultaneously"""
    
    def __init__(self):
        self.processor = DeepEEGProcessor()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def load_and_prepare_data(self, ictal_path, preictal_path, sample_rate=40, sequence_length=64):
        """Load and prepare data for deep learning"""
        from eeg_migraine_classifier import EEGDataProcessor
        
        print("\n" + "="*70)
        print("STEP 1: LOADING AND PREPARING DATA")
        print("="*70)
        
        # Use base processor for loading and cleaning
        processor = EEGDataProcessor()
        
        # Load data with sampling
        ictal_df, preictal_df = processor.load_data(ictal_path, preictal_path, sample_rate=sample_rate)
        
        # Clean data
        clean_data, labels, channels = processor.clean_data(ictal_df, preictal_df)
        
        print(f"\nPreparing deep learning data...")
        print(f"  Sequence length: {sequence_length}")
        
        # Prepare data for deep learning
        X_flat, X_sequences, y_flat, y_sequences, seq_len = self.processor.prepare_deep_learning_data(
            clean_data, labels, sequence_length=sequence_length
        )
        
        print(f"  Flat data shape: {X_flat.shape}")
        print(f"  Sequence data shape: {X_sequences.shape}")
        print(f"  Target shape: {y_flat.shape} (flat), {y_sequences.shape} (sequences)")
        
        return X_flat, X_sequences, y_flat, y_sequences, seq_len, clean_data, labels
    
    def train_models(self, X_flat, X_sequences, y_flat, y_sequences, sequence_length, 
                    epochs=100, batch_size=128):
        """Train CNN-LSTM and Transformer models simultaneously"""
        print("\n" + "="*70)
        print("STEP 2: TRAINING CNN-LSTM AND TRANSFORMER MODELS")
        print("="*70)
        
        num_features = X_flat.shape[1]
        sequence_shape = (sequence_length, num_features)
        
        # Ensure data is numpy arrays
        if hasattr(X_sequences, 'numpy'):
            X_sequences = X_sequences.numpy()
        if hasattr(y_sequences, 'numpy'):
            y_sequences = y_sequences.numpy()
        
        X_sequences = np.array(X_sequences, dtype=np.float32)
        y_sequences = np.array(y_sequences, dtype=np.float32)
        
        # Split data
        try:
            y_bins = np.digitize(y_sequences, bins=[0.0, 0.3, 0.7, 1.0])
            unique_bins, counts = np.unique(y_bins, return_counts=True)
            min_samples_per_bin = counts.min()
            if min_samples_per_bin >= 2:
                X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
                    X_sequences, y_sequences, test_size=0.3, random_state=42, shuffle=True, stratify=y_bins
                )
            else:
                X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
                    X_sequences, y_sequences, test_size=0.3, random_state=42, shuffle=True
                )
        except:
            X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
                X_sequences, y_sequences, test_size=0.3, random_state=42, shuffle=True
            )
        
        # Ensure train/test splits are numpy arrays
        X_seq_train = np.array(X_seq_train, dtype=np.float32)
        X_seq_test = np.array(X_seq_test, dtype=np.float32)
        y_seq_train = np.array(y_seq_train, dtype=np.float32)
        y_seq_test = np.array(y_seq_test, dtype=np.float32)
        
        print(f"\nTraining data: {X_seq_train.shape[0]:,} sequences")
        print(f"Test data: {X_seq_test.shape[0]:,} sequences")
        print(f"Target range: [{y_seq_train.min():.2f}, {y_seq_train.max():.2f}]")
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True, verbose=1, min_delta=1e-6
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1
        )
        
        # Compile settings
        optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        loss = 'mse'
        metrics = ['mae']
        
        def get_callbacks(model_name):
            safe_name = model_name.lower().replace('-', '_').replace(' ', '_')
            checkpoint = callbacks.ModelCheckpoint(
                f'best_{safe_name}_model.keras', 
                monitor='val_loss', 
                save_best_only=True, 
                verbose=0
            )
            return [early_stopping, reduce_lr, checkpoint]
        
        # Train CNN-LSTM Model
        print("\n" + "="*60)
        print("Training CNN-LSTM Hybrid Model...")
        print("="*60)
        print("Architecture: CNN (spatial) → LSTM (temporal) → Dense")
        print("Expected: Best performance, captures both spatial and temporal patterns")
        
        cnn_lstm_model = CNN_LSTM_Model(sequence_shape, num_features)
        cnn_lstm_model.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        start_time = time.time()
        history_cnn_lstm = cnn_lstm_model.model.fit(
            X_seq_train, y_seq_train,
            validation_data=(X_seq_test, y_seq_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=get_callbacks('CNN-LSTM'),
            verbose=1
        )
        cnn_lstm_time = time.time() - start_time
        
        cnn_lstm_pred = cnn_lstm_model.model.predict(X_seq_test, verbose=0)
        if hasattr(cnn_lstm_pred, 'numpy'):
            cnn_lstm_pred = cnn_lstm_pred.numpy()
        cnn_lstm_pred = np.array(cnn_lstm_pred, dtype=np.float32).flatten()
        
        cnn_lstm_r2 = r2_score(y_seq_test, cnn_lstm_pred)
        cnn_lstm_rmse = np.sqrt(mean_squared_error(y_seq_test, cnn_lstm_pred))
        cnn_lstm_mae = mean_absolute_error(y_seq_test, cnn_lstm_pred)
        
        self.models['CNN-LSTM'] = cnn_lstm_model
        self.results['CNN-LSTM'] = {
            'model': cnn_lstm_model.model,
            'r2': cnn_lstm_r2,
            'rmse': cnn_lstm_rmse,
            'mae': cnn_lstm_mae,
            'predictions': cnn_lstm_pred,
            'history': history_cnn_lstm,
            'training_time': cnn_lstm_time
        }
        
        print(f"\nCNN-LSTM Results:")
        print(f"  R² Score: {cnn_lstm_r2:.4f} ({cnn_lstm_r2*100:.1f}%)")
        print(f"  RMSE: {cnn_lstm_rmse:.4f}")
        print(f"  MAE: {cnn_lstm_mae:.4f}")
        print(f"  Training Time: {cnn_lstm_time/60:.1f} minutes")
        
        # Train Transformer Model
        print("\n" + "="*60)
        print("Training Transformer Model...")
        print("="*60)
        print("Architecture: Multi-Head Attention → Feed-Forward → Dense")
        print("Expected: Good performance, learns long-range dependencies")
        
        transformer_model = Transformer_Model(sequence_shape, num_features)
        transformer_model.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        start_time = time.time()
        history_transformer = transformer_model.model.fit(
            X_seq_train, y_seq_train,
            validation_data=(X_seq_test, y_seq_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=get_callbacks('Transformer'),
            verbose=1
        )
        transformer_time = time.time() - start_time
        
        transformer_pred = transformer_model.model.predict(X_seq_test, verbose=0)
        if hasattr(transformer_pred, 'numpy'):
            transformer_pred = transformer_pred.numpy()
        transformer_pred = np.array(transformer_pred, dtype=np.float32).flatten()
        
        transformer_r2 = r2_score(y_seq_test, transformer_pred)
        transformer_rmse = np.sqrt(mean_squared_error(y_seq_test, transformer_pred))
        transformer_mae = mean_absolute_error(y_seq_test, transformer_pred)
        
        self.models['Transformer'] = transformer_model
        self.results['Transformer'] = {
            'model': transformer_model.model,
            'r2': transformer_r2,
            'rmse': transformer_rmse,
            'mae': transformer_mae,
            'predictions': transformer_pred,
            'history': history_transformer,
            'training_time': transformer_time
        }
        
        print(f"\nTransformer Results:")
        print(f"  R² Score: {transformer_r2:.4f} ({transformer_r2*100:.1f}%)")
        print(f"  RMSE: {transformer_rmse:.4f}")
        print(f"  MAE: {transformer_mae:.4f}")
        print(f"  Training Time: {transformer_time/60:.1f} minutes")
        
        # Compare models
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        print(f"{'Model':<15} {'R² Score':<12} {'RMSE':<10} {'MAE':<10} {'Time (min)':<12}")
        print("-" * 70)
        for name, result in self.results.items():
            print(f"{name:<15} {result['r2']:>8.4f} ({result['r2']*100:>5.1f}%)  {result['rmse']:>8.4f}  {result['mae']:>8.4f}  {result['training_time']/60:>10.1f}")
        
        # Find best model
        best_name = max(self.results, key=lambda x: self.results[x]['r2'])
        self.best_model = self.models[best_name]
        self.best_model_name = best_name
        
        print(f"\nBest Model: {self.best_model_name} (R² = {self.results[self.best_model_name]['r2']:.4f})")
        
        return self.results
    
    def save_models(self, filepath_prefix='eeg_cnn_lstm_transformer'):
        """Save trained models"""
        import pickle
        
        print("\n" + "="*70)
        print("SAVING MODELS")
        print("="*70)
        
        # Save best model
        best_model_path = f'{filepath_prefix}_best_model.keras'
        self.best_model.model.save(best_model_path)
        print(f"Saved best model ({self.best_model_name}) to '{best_model_path}'")
        
        # Save all models
        for name, model_wrapper in self.models.items():
            safe_name = name.lower().replace('-', '_').replace(' ', '_')
            model_path = f'{filepath_prefix}_{safe_name}_model.keras'
            model_wrapper.model.save(model_path)
            print(f"Saved {name} model to '{model_path}'")
        
        # Save processor
        processor_path = f'{filepath_prefix}_processor.pkl'
        with open(processor_path, 'wb') as f:
            pickle.dump(self.processor, f)
        print(f"Saved processor to '{processor_path}'")
        
        # Save results
        results_path = f'{filepath_prefix}_results.pkl'
        results_to_save = {k: {key: val for key, val in v.items() if key != 'history'} 
                          for k, v in self.results.items()}
        with open(results_path, 'wb') as f:
            pickle.dump(results_to_save, f)
        print(f"Saved results to '{results_path}'")

def main():
    print("="*70)
    print("CNN-LSTM AND TRANSFORMER MODELS - COMPARISON TRAINING")
    print("="*70)
    print("\nThis script trains two advanced deep learning models:")
    print("  1. CNN-LSTM - Hybrid model (CNN for spatial + LSTM for temporal)")
    print("  2. Transformer - Multi-head attention model")
    print("\nData: chbmit_ictal_raw_data.csv and chbmit_preictal_raw_data.csv")
    print("="*70)
    
    start_time = time.time()
    
    # Initialize pipeline
    pipeline = CNN_LSTM_Transformer_Pipeline()
    
    # Get paths relative to model directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.dirname(script_dir)
    ictal_path = os.path.join(model_dir, 'chbmit_ictal_raw_data.csv')
    preictal_path = os.path.join(model_dir, 'chbmit_preictal_raw_data.csv')
    
    # Verify files exist
    if not os.path.exists(ictal_path):
        raise FileNotFoundError(f"Ictal data file not found: {ictal_path}")
    if not os.path.exists(preictal_path):
        raise FileNotFoundError(f"Preictal data file not found: {preictal_path}")
    
    print(f"\nData files:")
    print(f"  Ictal: {ictal_path}")
    print(f"  Preictal: {preictal_path}")
    
    # Load and prepare data
    X_flat, X_sequences, y_flat, y_sequences, sequence_length, clean_data, labels = pipeline.load_and_prepare_data(
        ictal_path=ictal_path,
        preictal_path=preictal_path,
        sample_rate=40,  # ~100k samples
        sequence_length=64  # 64 time steps per sequence
    )
    
    # Train models
    results = pipeline.train_models(
        X_flat, X_sequences, y_flat, y_sequences, sequence_length,
        epochs=100,  # Maximum epochs
        batch_size=128  # Batch size
    )
    
    # Save models
    pipeline.save_models()
    
    elapsed_time = time.time() - start_time
    hours = elapsed_time / 3600
    minutes = (elapsed_time % 3600) / 60
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nTotal training time: {hours:.1f} hours {minutes:.1f} minutes")
    print(f"\nFinal Results:")
    for name, result in results.items():
        print(f"  {name:15s}: R² = {result['r2']:.4f} ({result['r2']*100:.1f}%)")
    
    best_r2 = pipeline.results[pipeline.best_model_name]['r2']
    print(f"\nBest Model: {pipeline.best_model_name}")
    print(f"  R² Score: {best_r2:.4f} ({best_r2*100:.1f}%)")
    
    if best_r2 >= 0.75:
        print(f"\n✓ SUCCESS! Target of 75%+ accuracy achieved!")
        print(f"  Achieved: {best_r2*100:.1f}%")
    else:
        print(f"\n⚠ Target not yet reached. Current: {best_r2*100:.1f}%")
        print(f"  To improve further, try:")
        print(f"    - Increase epochs to 150-200")
        print(f"    - Increase sequence_length to 128")
        print(f"    - Decrease sample_rate to 20 (more data)")
    
    print("="*70)
    
    return pipeline, results

if __name__ == "__main__":
    try:
        pipeline, results = main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

