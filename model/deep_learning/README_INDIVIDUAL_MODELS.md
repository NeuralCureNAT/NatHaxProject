# Individual Deep Learning Model Training Files

## Overview

This directory contains 4 separate training scripts, one for each deep learning model:
1. **CNN1D** - `train_cnn1d.py`
2. **LSTM** - `train_lstm.py`
3. **CNN-LSTM** - `train_cnn_lstm.py`
4. **Transformer** - `train_transformer.py`

Each file is **completely independent** and trains only its specific model.

## Quick Start

### Train CNN1D Model
```bash
cd MigroMinder/model/deep_learning
python train_cnn1d.py
```

### Train LSTM Model
```bash
python train_lstm.py
```

### Train CNN-LSTM Model
```bash
python train_cnn_lstm.py
```

### Train Transformer Model
```bash
python train_transformer.py
```

## Files Overview

### 1. train_cnn1d.py
- **Model**: 1D Convolutional Neural Network
- **Architecture**: Conv1D → BatchNorm → Pooling → Dense
- **Best For**: Spatial pattern recognition in time-series
- **Expected R²**: 70-80%
- **Training Time**: 30-45 minutes
- **Output**: `eeg_cnn1d_model.keras`, `eeg_cnn1d_processor.pkl`

### 2. train_lstm.py
- **Model**: Long Short-Term Memory Network
- **Architecture**: LSTM → LSTM → LSTM → Dense
- **Best For**: Temporal sequence learning
- **Expected R²**: 65-75%
- **Training Time**: 45-60 minutes
- **Output**: `eeg_lstm_model.keras`, `eeg_lstm_processor.pkl`

### 3. train_cnn_lstm.py
- **Model**: Hybrid CNN-LSTM Model
- **Architecture**: CNN (spatial) → LSTM (temporal) → Dense
- **Best For**: Both spatial and temporal patterns
- **Expected R²**: 75-85% ⭐ (Best performance)
- **Training Time**: 60-90 minutes
- **Output**: `eeg_cnn_lstm_model.keras`, `eeg_cnn_lstm_processor.pkl`

### 4. train_transformer.py
- **Model**: Multi-Head Attention Transformer
- **Architecture**: Multi-Head Attention → Feed-Forward → Dense
- **Best For**: Long-range dependencies
- **Expected R²**: 70-80%
- **Training Time**: 90-120 minutes
- **Output**: `eeg_transformer_model.keras`, `eeg_transformer_processor.pkl`

## Data Files

All scripts use:
- **Ictal**: `chbmit_ictal_raw_data.csv` (1M+ rows)
- **Preictal**: `chbmit_preictal_raw_data.csv` (3M+ rows)

## Configuration

Each script can be customized by editing the `main()` function:

```python
# In main() function:
sample_rate=40,        # ~100k samples (increase for more data)
sequence_length=64,    # Time steps per sequence
epochs=100,            # Training epochs
batch_size=128         # Batch size
```

## Output Files

Each script generates:
- `eeg_[model]_model.keras` - Trained model
- `eeg_[model]_processor.pkl` - Data processor
- `best_[model]_model.keras` - Best model checkpoint

## Running Multiple Models

You can run all 4 models simultaneously in separate terminals:

**Terminal 1:**
```bash
python train_cnn1d.py
```

**Terminal 2:**
```bash
python train_lstm.py
```

**Terminal 3:**
```bash
python train_cnn_lstm.py
```

**Terminal 4:**
```bash
python train_transformer.py
```

## Comparing Results

After training all models, compare the results:

| Model | R² Score | RMSE | MAE | Time |
|-------|----------|------|-----|------|
| CNN1D | ? | ? | ? | ? |
| LSTM | ? | ? | ? | ? |
| CNN-LSTM | ? | ? | ? | ? |
| Transformer | ? | ? | ? | ? |

## Advantages of Separate Files

1. ✅ **Independent Training**: Train one model without others
2. ✅ **Easy Comparison**: Run models in parallel
3. ✅ **Focused Learning**: Study each architecture separately
4. ✅ **Flexible Configuration**: Different settings per model
5. ✅ **No Interference**: Original files remain unchanged

## Tips

1. **Start with CNN-LSTM**: Usually gives best results (75-85% R²)
2. **Compare Training Times**: See which model trains faster
3. **Adjust Parameters**: Each model can have different settings
4. **Run in Parallel**: Train multiple models simultaneously
5. **Check Results**: Compare R² scores to find best model

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

