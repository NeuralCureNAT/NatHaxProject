# Complete Workflow Guide

## What the Pipeline Does Automatically

When you run `python run_pipeline.py`, the system automatically:

### Step 1: Data Loading & Cleaning ✅
- Loads your EEG data from `raw_ictal.xlsx` and `raw_preictal.xlsx`
- Cleans missing values and aligns channels
- Creates non-ictal (baseline) data
- Converts labels to severity scores (0.0, 0.5, 1.0)

### Step 2: Initial Model Training & Testing ✅
- **Splits data**: 70% training, 30% testing (automatic)
- **Trains 4 regression models** on training data:
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - SVR (Support Vector Regressor)
  - Neural Network (MLP Regressor)
- **Tests all models** on test data
- **Evaluates performance** using RMSE, MAE, R² metrics
- **Selects best model** (lowest RMSE)
- **Shows results** with detailed metrics

### Step 3: Data Augmentation ✅
- Generates synthetic EEG data using VAE/GAN
- Augments each severity level to target number of samples
- Creates balanced dataset

### Step 4: Augmented Model Training & Testing ✅
- **Splits augmented data**: 70% training, 30% testing (automatic)
- **Trains 4 regression models** on augmented training data
- **Tests all models** on augmented test data
- **Evaluates performance** using RMSE, MAE, R² metrics
- **Selects best model** (lowest RMSE)
- **Shows results** with detailed metrics

### Step 5: Comparison ✅
- Compares initial vs augmented model performance
- Shows improvement metrics
- Displays detailed comparison reports

### Step 6: Save Models ✅
- Saves best augmented model to `eeg_migraine_augmented_model.pkl`
- Saves initial model to `eeg_migraine_initial_model.pkl`
- Saves processor to `eeg_migraine_processor.pkl`

## What You Need to Do

### 1. Prepare Your Data
Make sure you have:
- `raw_ictal.xlsx` - Ictal stage EEG data
- `raw_preictal.xlsx` - Preictal stage EEG data
- Files should be in the `model/` directory

### 2. Run the Pipeline
```bash
cd model
python run_pipeline.py
```

That's it! The pipeline does everything else automatically.

### 3. Check Results
After running, you'll see:
- Training progress for each model
- Test results with RMSE, MAE, R² scores
- Comparison between initial and augmented models
- Model performance metrics

### 4. Use Trained Models for Predictions
```bash
# Predict on new data
python predict.py your_new_data.xlsx

# Or test on existing data
python predict.py
```

## Understanding the Output

### Training Phase
```
Training Random Forest...
  MSE: 0.0123
  MAE: 0.0987
  RMSE: 0.1109
  R² Score: 0.8521
```

### Testing Phase (Automatic)
The same metrics are calculated on the **test set** (30% of data held out)
- Lower RMSE/MAE = Better predictions
- Higher R² = Better model fit (closer to 1.0 is better)

### Comparison
```
Initial Model:
  RMSE: 0.1500
  MAE: 0.1200
  R² Score: 0.8000

Augmented Model:
  RMSE: 0.1200  (20% improvement)
  MAE: 0.0950   (21% improvement)
  R² Score: 0.8500  (6% improvement)
```

## File Structure After Running

```
model/
├── eeg_migraine_initial_model.pkl      # Initial trained model
├── eeg_migraine_augmented_model.pkl    # Best model (recommended)
├── eeg_migraine_processor.pkl          # Data processor
└── ... (other files)
```

## Next Steps After Training

### Option 1: Make Predictions on New Data
```bash
python predict.py new_patient_data.xlsx
```

### Option 2: Evaluate Model Performance
Check the output metrics:
- **RMSE < 0.15**: Good performance
- **R² > 0.8**: Good model fit
- **MAE < 0.12**: Low prediction error

### Option 3: Improve Model (if needed)
If performance is not good enough:
1. Increase `target_samples` in `run_pipeline.py` (e.g., 200 → 500)
2. Increase `augmentation_epochs` (e.g., 100 → 200)
3. Collect more real data
4. Try different augmentation methods (VAE vs GAN)

## Complete Example Workflow

```bash
# 1. Navigate to model directory
cd model

# 2. Run complete pipeline (trains AND tests automatically)
python run_pipeline.py

# Output:
# - Training progress
# - Test results for each model
# - Best model selection
# - Augmentation progress
# - Final comparison
# - Model files saved

# 3. Make predictions on new data
python predict.py new_data.xlsx

# Output:
# - Severity scores for each sample
# - Predicted stage labels
# - Interpretation of results
```

## Important Notes

1. **Automatic Train/Test Split**: The pipeline automatically splits your data into training (70%) and testing (30%) sets
2. **No Manual Testing Required**: Testing happens automatically during training
3. **Models Are Saved**: After training, models are saved and ready to use
4. **Regression Mode**: All models predict continuous severity scores (0.0-1.0)
5. **Best Model Selected**: The system automatically selects the best performing model

## Troubleshooting

### If Training Fails:
- Check that data files exist
- Verify data format is correct
- Check error messages for specific issues

### If Performance is Poor:
- Increase augmentation samples
- Increase training epochs
- Check data quality
- Verify data preprocessing

### If Predictions Are Wrong:
- Make sure you're using the augmented model
- Check that new data has same format as training data
- Verify all required channels are present

## Summary

**You just need to:**
1. ✅ Put your data files in `model/` directory
2. ✅ Run `python run_pipeline.py`
3. ✅ Wait for training to complete (it tests automatically)
4. ✅ Use saved models for predictions

**The pipeline automatically:**
- ✅ Trains models
- ✅ Tests models
- ✅ Compares performance
- ✅ Saves best models
- ✅ Shows detailed results

No manual testing or evaluation needed - it's all automated! 🚀

