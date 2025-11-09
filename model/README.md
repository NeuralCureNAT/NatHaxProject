# EEG Migraine Severity Regression System

This project implements a complete pipeline for predicting continuous migraine severity scores from EEG data:
- **Severity Score**: 0.0 to 1.0 (continuous scale)
  - **0.0** = Non-Ictal (baseline/healthy)
  - **0.5** = Preictal (pre-migraine)
  - **1.0** = Ictal (active migraine)

## Features

1. **Data Cleaning**: Handles messy EEG data with missing values and inconsistent channels
2. **Initial Regression Model Training**: Trains multiple regression models (Random Forest, Gradient Boosting, SVR, Neural Network)
3. **Generative Data Augmentation**: Uses VAE (Variational Autoencoder) or GAN to generate synthetic EEG data
4. **Improved Model Training**: Retrains models on augmented dataset for better performance
5. **Performance Comparison**: Compares initial vs augmented model performance using regression metrics (RMSE, MAE, R²)

## Installation

### For Linux/macOS Users

1. **Clone or download the repository:**
```bash
git clone <repository-url>
cd nathax
```

2. **Create a virtual environment (in the parent directory):**
```bash
cd ..  # Go to parent directory (nathax)
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**
```bash
cd model  # Go to model directory
pip install -r requirements.txt
```

### For Windows Users

1. **Clone or download the repository:**
   - Open Command Prompt or PowerShell
   - Navigate to where you want to clone the project
   - Run: `git clone <repository-url>`
   - Or download and extract the ZIP file

2. **Create a virtual environment (in the parent directory):**
```cmd
cd nathax
cd ..
python -m venv venv
venv\Scripts\activate
```
   **Note:** If you get an execution policy error in PowerShell, run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Install dependencies:**
```cmd
cd model
pip install -r requirements.txt
```

**Alternative for Windows (using Python Launcher):**
```cmd
cd nathax
cd ..
py -3 -m venv venv
venv\Scripts\activate
cd model
pip install -r requirements.txt
```

## Usage

### Quick Start

**Note:** All files are in the `model/` directory. Navigate to it first:

**Linux/macOS:**
```bash
cd model
```

**Windows:**
```cmd
cd model
```

### Step 1: Train the Model (Trains AND Tests Automatically)

```bash
# Linux/macOS/Windows (all use same command)
python run_pipeline.py
```

**What this does automatically:**
- ✅ Loads and cleans your data
- ✅ Trains regression models (70% of data)
- ✅ Tests models automatically (30% of data)
- ✅ Generates synthetic data
- ✅ Retrains on augmented data
- ✅ Tests augmented models automatically
- ✅ Compares performance
- ✅ Saves best model

**You'll see:**
- Training progress for each model
- Test results with RMSE, MAE, R² scores
- Best model selection
- Performance comparison

### Step 2: Make Predictions

```bash
# Predict on new data
python predict.py your_new_data.xlsx

# Or test on training data
python predict.py
```

### For Better Quality (Slower)

```bash
python improve_augmentation.py
```

**Windows Note:** If `python` doesn't work, try `py` or `python3`:
```cmd
py run_pipeline.py
```

### What Happens During Training?

1. **Data Split**: Automatically splits into 70% training, 30% testing
2. **Model Training**: Trains 4 regression models on training data
3. **Automatic Testing**: Tests all models on test data (you don't need to do anything)
4. **Model Selection**: Automatically selects best model (lowest RMSE)
5. **Results Display**: Shows RMSE, MAE, R² scores for each model
6. **Model Saving**: Saves best model to `.pkl` files

**No manual testing required - it's all automatic!**

### Custom Configuration

Edit `run_pipeline.py` to customize:
- `target_samples`: Target number of samples per class 
  - Fast testing: 50
  - Good quality: 100-200
  - Production: 300-500
- `augmentation_method`: 'vae' or 'gan' (default: 'vae')
  - VAE: Better for small datasets, more stable
  - GAN: Better for larger datasets, may need more tuning
- `augmentation_epochs`: Training epochs for generator
  - Fast testing: 30-50
  - Good quality: 100-150
  - Production: 200-300

### Individual Components

#### Data Processing Only
```python
from eeg_migraine_classifier import EEGDataProcessor

processor = EEGDataProcessor()
ictal_df, preictal_df = processor.load_data('raw_ictal.xlsx', 'raw_preictal.xlsx')
clean_data, labels, channels = processor.clean_data(ictal_df, preictal_df)
# Labels are converted to severity scores: nonictal=0.0, preictal=0.5, ictal=1.0
```

#### Data Augmentation Only
```python
from generative_data_augmentation import EEGDataAugmenter
import numpy as np

augmenter = EEGDataAugmenter(method='vae')
augmenter.fit(X_train, epochs=100)
synthetic_data = augmenter.generate(n_samples=100)
```

## File Structure

- `eeg_migraine_classifier.py`: Data processing and classification models
- `generative_data_augmentation.py`: VAE and GAN for synthetic data generation
- `main_pipeline.py`: Complete pipeline integration
- `run_pipeline.py`: Main execution script
- `predict.py`: Prediction script for new EEG data
- `improve_augmentation.py`: Enhanced augmentation settings
- `raw_ictal.xlsx`: Ictal stage EEG data
- `raw_preictal.xlsx`: Preictal stage EEG data
- `eeg_migraine_initial_model.pkl`: Initial trained model
- `eeg_migraine_augmented_model.pkl`: Enhanced model (recommended)
- `eeg_migraine_processor.pkl`: Data processor

## Output

The pipeline generates:
- `eeg_migraine_initial_model.pkl`: Initial trained model
- `eeg_migraine_augmented_model.pkl`: Enhanced model trained on augmented data
- `eeg_migraine_processor.pkl`: Data processor for future predictions

## Model Performance

The system trains and compares multiple regression models:
- **Random Forest Regressor**: Ensemble method with good generalization
- **Gradient Boosting Regressor**: Sequential ensemble with strong performance
- **Support Vector Regressor (SVR)**: Kernel-based method for non-linear relationships
- **Neural Network (MLP Regressor)**: Deep learning approach with multiple hidden layers

Best performing model is automatically selected based on **lowest RMSE** (Root Mean Squared Error).

### Performance Metrics

- **RMSE (Root Mean Squared Error)**: Lower is better, measures average prediction error
- **MAE (Mean Absolute Error)**: Lower is better, measures average absolute deviation
- **R² Score**: Higher is better (0-1), measures how well model explains variance
- **Prediction Range**: Shows min/max predicted severity scores

### Performance Notes

- Regression models predict continuous severity scores (0.0-1.0)
- Augmented model uses larger, more realistic test sets
- Lower RMSE/MAE on augmented test set indicates better performance
- Increase augmentation epochs and samples for better synthetic data quality
- The system includes statistical fallback for very small datasets
- Predictions can be converted back to stage labels (nonictal/preictal/ictal) using thresholds

## Data Augmentation

For small datasets, the system uses:
1. **VAE (Variational Autoencoder)**: Learns latent representations and generates new samples
2. **GAN (Generative Adversarial Network)**: Creates realistic synthetic data through adversarial training
3. **Statistical Fallback**: Uses Gaussian distribution if deep learning methods fail

## Notes

- The system handles missing values and aligns channels automatically
- Non-ictal (baseline) class is generated from preictal data statistics
- Labels are converted to continuous severity scores: nonictal=0.0, preictal=0.5, ictal=1.0
- Predictions are continuous severity scores that can be converted to stage labels
- Small datasets automatically use statistical augmentation as fallback
- Increase `augmentation_epochs` for better quality synthetic data (150-200 recommended)
- Regression allows for more nuanced predictions than classification (e.g., 0.3 = mild preictal, 0.7 = severe preictal)

## Requirements

- Python 3.7+ (Python 3.8+ recommended)
- TensorFlow 2.13+
- scikit-learn
- pandas
- numpy
- openpyxl (for Excel file reading)

## Platform Compatibility

✅ **This project is cross-platform compatible:**
- ✅ Windows 10/11
- ✅ Linux (Ubuntu, Debian, etc.)
- ✅ macOS
- ✅ Works with Python 3.7, 3.8, 3.9, 3.10, 3.11, 3.12+

## Troubleshooting

### Windows-Specific Issues

1. **"python is not recognized" error:**
   - Use `py` instead of `python`: `py run_pipeline.py`
   - Or add Python to your PATH environment variable
   - Or use full path: `C:\Python39\python.exe run_pipeline.py`

2. **PowerShell execution policy error:**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. **Virtual environment activation fails:**
   - Make sure you're in the correct directory
   - Try: `.\venv\Scripts\activate` (with backslash and dot)
   - Or use: `venv\Scripts\activate.bat` in Command Prompt

4. **TensorFlow installation issues on Windows:**
   - Make sure you have Visual C++ Redistributable installed
   - Try: `pip install --upgrade pip` first
   - Then: `pip install tensorflow`

5. **File path issues:**
   - Use forward slashes `/` in paths (Python handles this automatically)
   - Or use raw strings: `r'C:\path\to\file'`
   - The code uses relative paths, so this shouldn't be an issue

### General Issues

1. **ModuleNotFoundError:**
   - Make sure virtual environment is activated
   - Reinstall dependencies: `pip install -r requirements.txt --upgrade`

2. **Out of memory errors:**
   - Reduce `target_samples` in `run_pipeline.py`
   - Reduce `augmentation_epochs`
   - Close other applications

3. **Excel file reading errors:**
   - Make sure `openpyxl` is installed: `pip install openpyxl`
   - Check that Excel files are not corrupted
   - Verify file paths are correct

### Getting Help

If you encounter issues:
1. Check that all dependencies are installed correctly
2. Verify Python version: `python --version` (should be 3.7+)
3. Make sure you're in the `model/` directory when running scripts
4. Check that virtual environment is activated
5. Review error messages carefully - they usually indicate the problem

## File Paths

The code uses relative paths and is designed to work cross-platform:
- All file paths use forward slashes (work on Windows, Linux, macOS)
- Scripts expect to be run from the `model/` directory
- Data files should be in the same directory as scripts
- Models are saved in the same directory

## Additional Resources

- **Windows Users:** See `INSTALL_WINDOWS.md` for detailed Windows-specific installation instructions
- **Quick Start:** See `QUICKSTART.md` for a quick reference guide
- **Troubleshooting:** See the Troubleshooting section above for common issues
- **Platform Support:** This project is fully cross-platform compatible

## License

This project is for research and educational purposes.

