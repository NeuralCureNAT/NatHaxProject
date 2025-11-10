# CNN-LSTM and Transformer Comparison Training

## Overview

This script trains and compares two advanced deep learning models:
1. **CNN-LSTM** - Hybrid model combining CNN (spatial features) and LSTM (temporal features)
2. **Transformer** - Multi-head attention model for learning long-range dependencies

## Features

- ✅ Uses `chbmit_ictal_raw_data.csv` and `chbmit_preictal_raw_data.csv`
- ✅ Trains both models simultaneously for comparison
- ✅ Shows training time for each model
- ✅ Detailed performance metrics (R², RMSE, MAE)
- ✅ Side-by-side comparison table
- ✅ Saves both models for future use

## Quick Start

```bash
cd MigroMinder/model/deep_learning
python train_cnn_lstm_transformer.py
```

## What It Does

1. **Loads Data**: Reads from `chbmit_ictal_raw_data.csv` and `chbmit_preictal_raw_data.csv`
2. **Samples Data**: Uses `sample_rate=40` (~100k samples from 4M rows)
3. **Creates Sequences**: Converts data to sequences of 64 time steps
4. **Trains CNN-LSTM**: Trains hybrid CNN-LSTM model
5. **Trains Transformer**: Trains Transformer model with multi-head attention
6. **Compares Results**: Shows side-by-side comparison
7. **Saves Models**: Saves both models and the best performing one

## Model Architectures

### CNN-LSTM Model
- **Architecture**: CNN → LSTM → Dense
- **Strengths**: 
  - Captures spatial patterns (CNN)
  - Learns temporal sequences (LSTM)
  - Best for time-series with spatial structure
- **Expected Performance**: 75-85% R² Score
- **Training Time**: 60-90 minutes

### Transformer Model
- **Architecture**: Multi-Head Attention → Feed-Forward → Dense
- **Strengths**:
  - Learns long-range dependencies
  - Parallel processing (faster training)
  - Attention mechanism
- **Expected Performance**: 70-80% R² Score
- **Training Time**: 90-120 minutes

## Configuration

Edit `train_cnn_lstm_transformer.py` to adjust:

```python
# In main() function:
sample_rate=40,        # ~100k samples (increase for more data)
sequence_length=64,    # Time steps per sequence (increase for longer context)
epochs=100,            # Training epochs
batch_size=128         # Batch size
```

## Output Files

After training, you'll get:
- `eeg_cnn_lstm_transformer_best_model.keras` - Best model (use this!)
- `eeg_cnn_lstm_transformer_cnn_lstm_model.keras` - CNN-LSTM model
- `eeg_cnn_lstm_transformer_transformer_model.keras` - Transformer model
- `eeg_cnn_lstm_transformer_processor.pkl` - Data processor
- `eeg_cnn_lstm_transformer_results.pkl` - Training results

## Example Output

```
======================================================================
MODEL COMPARISON
======================================================================
Model           R² Score     RMSE       MAE        Time (min)
----------------------------------------------------------------------
CNN-LSTM        0.7856 (78.6%)  0.2144  0.1856      85.3
Transformer     0.7623 (76.2%)  0.2377  0.2012      112.7

Best Model: CNN-LSTM (R² = 0.7856)
```

## Tips for Best Results

1. **More Data**: Decrease `sample_rate` to 20 (more data = better accuracy)
2. **Longer Sequences**: Increase `sequence_length` to 128 (better temporal context)
3. **More Epochs**: Increase `epochs` to 150-200 (better convergence)
4. **Compare Both**: Both models have different strengths, compare results!

## Understanding the Models

### CNN-LSTM
- **When to use**: When you have both spatial and temporal patterns
- **How it works**: 
  1. CNN extracts spatial features (patterns across channels)
  2. LSTM processes temporal sequences (patterns over time)
  3. Dense layers combine features for prediction

### Transformer
- **When to use**: When you need to learn long-range dependencies
- **How it works**:
  1. Multi-head attention learns relationships between all time steps
  2. Feed-forward networks process features
  3. Layer normalization stabilizes training
  4. Dense layers make final predictions

## Troubleshooting

**Memory Error?**
- Increase `sample_rate` to 60-80
- Decrease `batch_size` to 64
- Decrease `sequence_length` to 32

**Slow Training?**
- Increase `sample_rate` to 60
- Decrease `epochs` to 50
- Decrease `batch_size` to 64

**Low Accuracy?**
- Decrease `sample_rate` to 20 (more data)
- Increase `sequence_length` to 128
- Increase `epochs` to 150-200

## Differences from Full Pipeline

This script is focused on comparing only the two most advanced models:
- ✅ Faster training (only 2 models instead of 4)
- ✅ Direct comparison of CNN-LSTM vs Transformer
- ✅ Training time tracking for each model
- ✅ Uses chbmit CSV files directly
- ✅ Clear side-by-side comparison

The original `train_deep_learning.py` trains all 4 models (CNN1D, LSTM, CNN-LSTM, Transformer) for comprehensive comparison.

