# Deep Learning Pipeline - Quick Start

## 🚀 Run Deep Learning Pipeline

```bash
cd MigroMinder/model/deep_learning
python train_deep_learning.py
```

## 📊 What It Does

1. ✅ Loads and cleans data (~100k samples)
2. ✅ Creates sequences (64 time steps per sequence)
3. ✅ Trains 4 deep learning models:
   - CNN1D (spatial patterns)
   - LSTM (temporal sequences)
   - CNN-LSTM (spatial + temporal) ⭐ BEST
   - Transformer (attention mechanism)
4. ✅ Compares all models
5. ✅ Saves best model

## 🎯 Expected Results

- **Best Model**: CNN-LSTM Hybrid
- **R² Score**: 75-85%+
- **Training Time**: 3-4 hours
- **RMSE**: 0.12-0.18
- **MAE**: 0.10-0.15

## 📁 Output Files

After training:
- `eeg_deep_learning_best_model.keras` - Best model (use this!)
- `eeg_deep_learning_cnn_lstm_model.keras` - CNN-LSTM model
- `eeg_deep_learning_processor.pkl` - Data processor
- `eeg_deep_learning_results.pkl` - Training results

## ⚙️ Configuration

Edit `train_deep_learning.py` to adjust:

```python
sample_rate=40           # ~100k samples
sequence_length=64       # 64 time steps per sequence
epochs=100               # Training epochs
batch_size=128           # Batch size
```

## 🔍 Troubleshooting

### If accuracy is below 75%:
- Increase `sequence_length` to 128
- Increase `sample_rate` to 20 (more data)
- Increase `epochs` to 150

### If training is too slow:
- Reduce `sequence_length` to 32
- Reduce `sample_rate` to 60
- Reduce `epochs` to 50

### If out of memory:
- Reduce `batch_size` to 64
- Reduce `sequence_length` to 32
- Reduce `sample_rate` to 80

## 📚 More Information

See `README.md` for complete documentation.

---

**Target**: 75%+ R² Score
**Best Model**: CNN-LSTM Hybrid
**Status**: ✅ Ready to Use

