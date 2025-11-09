# Quick Guide: What Happens When You Run the Pipeline

## 🚀 Simple Answer

**YES, it trains with regression AND tests automatically!** You just need to run one command.

## 📋 Step-by-Step: What You Do

### Step 1: Make sure your data is ready
```bash
cd model
# Make sure raw_ictal.xlsx and raw_preictal.xlsx are in this directory
```

### Step 2: Run the pipeline
```bash
python run_pipeline.py
```

### Step 3: Wait for it to complete
The pipeline does EVERYTHING automatically:
- ✅ Loads your data
- ✅ Cleans your data
- ✅ Trains models (regression)
- ✅ Tests models (automatic)
- ✅ Generates synthetic data
- ✅ Retrains on augmented data
- ✅ Tests augmented models (automatic)
- ✅ Compares results
- ✅ Saves best model

### Step 4: Use the trained model
```bash
python predict.py your_new_data.xlsx
```

## 🔍 What Happens During Training & Testing

### Phase 1: Initial Model Training & Testing

```
1. Data is split automatically:
   - 70% → Training set
   - 30% → Test set

2. Models are trained on training set:
   - Random Forest Regressor
   - Gradient Boosting Regressor
   - SVR
   - Neural Network

3. Models are tested on test set (AUTOMATIC):
   - RMSE calculated
   - MAE calculated
   - R² score calculated
   - Best model selected

4. Results shown:
   Training Random Forest...
     MSE: 0.0123
     MAE: 0.0987
     RMSE: 0.1109
     R² Score: 0.8521
```

### Phase 2: Data Augmentation

```
1. Synthetic data generated for each severity level
2. Dataset expanded (e.g., 48 samples → 300 samples)
```

### Phase 3: Augmented Model Training & Testing

```
1. Augmented data split:
   - 70% → Training set
   - 30% → Test set

2. Models trained on augmented training set
3. Models tested on augmented test set (AUTOMATIC)
4. Best model selected
5. Results compared with initial model
```

### Phase 4: Model Saving

```
Saved models:
- eeg_migraine_augmented_model.pkl (BEST MODEL - use this!)
- eeg_migraine_initial_model.pkl
- eeg_migraine_processor.pkl
```

## 📊 What You'll See

### Training Output Example:
```
======================================================================
STEP 2: TRAINING INITIAL REGRESSION MODEL (Original Data)
======================================================================

Training set: 33 samples
Test set: 15 samples
Severity range: [0.00, 1.00]

============================================================
TRAINING REGRESSION MODELS
============================================================

Training Random Forest...
  MSE: 0.0123
  MAE: 0.0987
  RMSE: 0.1109
  R² Score: 0.8521
  Predicted range: [0.12, 0.98]
  Actual range: [0.00, 1.00]

Training Gradient Boosting...
  ...

============================================================
Best Model: Random Forest
  RMSE: 0.1109
  MAE: 0.0987
  R² Score: 0.8521
============================================================
```

### Testing is AUTOMATIC - You'll see:
- Test set metrics (RMSE, MAE, R²)
- Prediction vs actual comparison
- Model performance scores

## ✅ What You DON'T Need to Do

- ❌ Don't manually split data (automatic)
- ❌ Don't manually test models (automatic)
- ❌ Don't manually evaluate (automatic)
- ❌ Don't manually select best model (automatic)
- ❌ Don't manually save models (automatic)

## ✅ What You DO Need to Do

1. ✅ Put your data files in `model/` directory
2. ✅ Run `python run_pipeline.py`
3. ✅ Wait for completion
4. ✅ Use saved model for predictions

## 🎯 After Training is Complete

### Option 1: Make Predictions
```bash
python predict.py new_patient_data.xlsx
```

### Option 2: Check Model Performance
Look at the output metrics:
- **RMSE**: Lower is better (target: < 0.15)
- **MAE**: Lower is better (target: < 0.12)
- **R²**: Higher is better (target: > 0.8)

### Option 3: Improve Model (if needed)
Edit `run_pipeline.py`:
```python
target_samples=200,  # Increase for more synthetic data
augmentation_epochs=150  # Increase for better quality
```

## 📁 Files Created After Training

```
model/
├── eeg_migraine_augmented_model.pkl    ← USE THIS for predictions
├── eeg_migraine_initial_model.pkl
└── eeg_migraine_processor.pkl          ← Needed for preprocessing
```

## 🔄 Complete Workflow

```
1. Prepare Data
   └─> raw_ictal.xlsx
   └─> raw_preictal.xlsx

2. Run Pipeline
   └─> python run_pipeline.py
       ├─> Load & Clean Data
       ├─> Train Initial Model (70% train, 30% test)
       ├─> Test Initial Model (AUTOMATIC)
       ├─> Generate Synthetic Data
       ├─> Train Augmented Model (70% train, 30% test)
       ├─> Test Augmented Model (AUTOMATIC)
       ├─> Compare Results
       └─> Save Models

3. Use Trained Model
   └─> python predict.py new_data.xlsx
       └─> Get severity scores and predictions
```

## 💡 Key Points

1. **Regression Mode**: Yes, all models use regression (predict continuous scores 0.0-1.0)

2. **Automatic Testing**: Testing happens automatically during training - no manual testing needed

3. **Train/Test Split**: Data is automatically split 70/30 (training/test)

4. **Best Model Selected**: System automatically selects best model (lowest RMSE)

5. **Models Saved**: Best model is saved and ready to use

6. **Ready for Predictions**: After training, you can immediately use the model for predictions

## 🚨 Common Questions

**Q: Do I need to test manually?**
A: No! Testing is automatic. The pipeline tests on 30% of data automatically.

**Q: How do I know if the model is good?**
A: Check the RMSE, MAE, and R² scores in the output. Lower RMSE/MAE and higher R² = better model.

**Q: Which model should I use?**
A: Use `eeg_migraine_augmented_model.pkl` - it's the best model trained on augmented data.

**Q: Can I use the model right away?**
A: Yes! After training completes, you can use `predict.py` to make predictions.

**Q: What if performance is poor?**
A: Increase `target_samples` and `augmentation_epochs` in `run_pipeline.py`, then retrain.

## 📝 Summary

**To train and test your model:**
```bash
cd model
python run_pipeline.py
```

**That's it!** The pipeline does:
- ✅ Training (regression)
- ✅ Testing (automatic)
- ✅ Model selection (automatic)
- ✅ Model saving (automatic)

**Then use the model:**
```bash
python predict.py your_data.xlsx
```

No manual testing or evaluation needed - everything is automated! 🎉

