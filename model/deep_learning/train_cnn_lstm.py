"""
CNN-LSTM Model Training for EEG Migraine Severity Prediction
Hybrid CNN-LSTM Model combining spatial and temporal features
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
            # Input layer
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

def load_and_prepare_data(ictal_path, preictal_path, sample_rate=40, sequence_length=64):
    """Load and prepare data for deep learning"""
    from eeg_migraine_classifier import EEGDataProcessor
    
    print("\n" + "="*70)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("="*70)
    
    processor = EEGDataProcessor()
    ictal_df, preictal_df = processor.load_data(ictal_path, preictal_path, sample_rate=sample_rate)
    clean_data, labels, channels = processor.clean_data(ictal_df, preictal_df)
    
    print(f"\nPreparing deep learning data...")
    print(f"  Sequence length: {sequence_length}")
    
    dl_processor = DeepEEGProcessor()
    X_flat, X_sequences, y_flat, y_sequences, seq_len = dl_processor.prepare_deep_learning_data(
        clean_data, labels, sequence_length=sequence_length
    )
    
    print(f"  Flat data shape: {X_flat.shape}")
    print(f"  Sequence data shape: {X_sequences.shape}")
    print(f"  Target shape: {y_flat.shape} (flat), {y_sequences.shape} (sequences)")
    
    return X_sequences, y_sequences, seq_len, dl_processor

def train_cnn_lstm_model(X_sequences, y_sequences, sequence_length, epochs=100, batch_size=128):
    """Train CNN-LSTM model"""
    print("\n" + "="*70)
    print("STEP 2: TRAINING CNN-LSTM HYBRID MODEL")
    print("="*70)
    
    # Ensure data is numpy arrays
    if hasattr(X_sequences, 'numpy'):
        X_sequences = X_sequences.numpy()
    if hasattr(y_sequences, 'numpy'):
        y_sequences = y_sequences.numpy()
    
    X_sequences = np.array(X_sequences, dtype=np.float32)
    y_sequences = np.array(y_sequences, dtype=np.float32)
    
    num_features = X_sequences.shape[2]
    sequence_shape = (sequence_length, num_features)
    
    # Split data
    try:
        y_bins = np.digitize(y_sequences, bins=[0.0, 0.3, 0.7, 1.0])
        unique_bins, counts = np.unique(y_bins, return_counts=True)
        min_samples_per_bin = counts.min()
        if min_samples_per_bin >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X_sequences, y_sequences, test_size=0.3, random_state=42, shuffle=True, stratify=y_bins
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_sequences, y_sequences, test_size=0.3, random_state=42, shuffle=True
            )
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            X_sequences, y_sequences, test_size=0.3, random_state=42, shuffle=True
        )
    
    # Ensure train/test splits are numpy arrays
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    print(f"\nTraining data: {X_train.shape[0]:,} sequences")
    print(f"Test data: {X_test.shape[0]:,} sequences")
    print(f"Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
    
    # Build model
    print("\n" + "="*60)
    print("Building CNN-LSTM Hybrid Model...")
    print("="*60)
    print("Architecture: CNN (spatial) → LSTM (temporal) → Dense")
    print("Expected: Best performance, captures both spatial and temporal patterns")
    
    cnn_lstm_model = CNN_LSTM_Model(sequence_shape, num_features)
    
    # Compile model
    optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    cnn_lstm_model.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True, verbose=1, min_delta=1e-6
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1
    )
    checkpoint = callbacks.ModelCheckpoint(
        'best_cnn_lstm_model.keras', 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=0
    )
    
    # Train model
    print("\nTraining CNN-LSTM Model...")
    start_time = time.time()
    
    history = cnn_lstm_model.model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluate model
    y_pred = cnn_lstm_model.model.predict(X_test, verbose=0)
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    y_pred = np.array(y_pred, dtype=np.float32).flatten()
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n" + "="*70)
    print("CNN-LSTM MODEL RESULTS")
    print("="*70)
    print(f"R² Score: {r2:.4f} ({r2*100:.1f}%)")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print("="*70)
    
    return cnn_lstm_model, history, {'r2': r2, 'rmse': rmse, 'mae': mae, 'training_time': training_time}

def save_model(model, processor, filepath_prefix='eeg_cnn_lstm'):
    """Save trained model and processor"""
    import pickle
    
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    # Save model
    model_path = f'{filepath_prefix}_model.keras'
    model.model.save(model_path)
    print(f"Saved model to '{model_path}'")
    
    # Save processor
    processor_path = f'{filepath_prefix}_processor.pkl'
    with open(processor_path, 'wb') as f:
        pickle.dump(processor, f)
    print(f"Saved processor to '{processor_path}'")

def main():
    print("="*70)
    print("CNN-LSTM HYBRID MODEL TRAINING")
    print("="*70)
    print("\nHybrid CNN-LSTM Model for EEG Migraine Severity Prediction")
    print("Combines CNN (spatial features) and LSTM (temporal features)")
    print("Data: chbmit_ictal_raw_data.csv and chbmit_preictal_raw_data.csv")
    print("="*70)
    
    start_time = time.time()
    
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
    X_sequences, y_sequences, sequence_length, processor = load_and_prepare_data(
        ictal_path=ictal_path,
        preictal_path=preictal_path,
        sample_rate=40,  # ~100k samples
        sequence_length=64  # 64 time steps per sequence
    )
    
    # Train model
    model, history, results = train_cnn_lstm_model(
        X_sequences, y_sequences, sequence_length,
        epochs=100,  # Maximum epochs
        batch_size=128  # Batch size
    )
    
    # Save model
    save_model(model, processor)
    
    elapsed_time = time.time() - start_time
    hours = elapsed_time / 3600
    minutes = (elapsed_time % 3600) / 60
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nTotal time: {hours:.1f} hours {minutes:.1f} minutes")
    print(f"R² Score: {results['r2']:.4f} ({results['r2']*100:.1f}%)")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"MAE: {results['mae']:.4f}")
    
    if results['r2'] >= 0.75:
        print(f"\n✓ SUCCESS! Target of 75%+ accuracy achieved!")
    else:
        print(f"\n⚠ To improve, try:")
        print(f"    - Increase epochs to 150-200")
        print(f"    - Increase sequence_length to 128")
        print(f"    - Decrease sample_rate to 20 (more data)")
    
    print("="*70)
    
    return model, processor, results

if __name__ == "__main__":
    try:
        model, processor, results = main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

