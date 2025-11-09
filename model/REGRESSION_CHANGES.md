# Changes from Classification to Regression

## Overview

The system has been converted from **classification** to **regression** to predict continuous migraine severity scores instead of discrete stage classes.

## Key Changes

### 1. Target Variable
- **Before (Classification)**: Discrete classes (nonictal, preictal, ictal)
- **After (Regression)**: Continuous severity scores (0.0 to 1.0)
  - 0.0 = Non-Ictal (baseline/healthy)
  - 0.5 = Preictal (pre-migraine)
  - 1.0 = Ictal (active migraine)

### 2. Models
- **Before**: `RandomForestClassifier`, `GradientBoostingClassifier`, `SVC`, `MLPClassifier`
- **After**: `RandomForestRegressor`, `GradientBoostingRegressor`, `SVR`, `MLPRegressor`

### 3. Metrics
- **Before**: Accuracy, F1-Score, Classification Report
- **After**: RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), R² Score

### 4. Evaluation
- **Before**: Best model selected by highest accuracy
- **After**: Best model selected by lowest RMSE

### 5. Predictions
- **Before**: Predicted class labels (nonictal/preictal/ictal)
- **After**: Predicted severity scores (0.0-1.0) that can be converted to labels

## Benefits of Regression

1. **More Nuanced Predictions**: Can predict severity levels like 0.3 (mild preictal) or 0.7 (severe preictal)
2. **Continuous Scale**: Better represents the gradual nature of migraine progression
3. **Error Measurement**: RMSE/MAE provide more informative error metrics than accuracy
4. **Flexibility**: Can set custom thresholds for stage classification if needed

## Code Changes

### Files Modified
1. `eeg_migraine_classifier.py`
   - Changed `EEGMigraineClassifier` → `EEGMigraineRegressor`
   - Added severity mapping: `{'nonictal': 0.0, 'preictal': 0.5, 'ictal': 1.0}`
   - Added `severity_to_label()` method for converting scores back to labels
   - Updated metrics to RMSE, MAE, R²

2. `main_pipeline.py`
   - Updated pipeline to use regression models
   - Changed evaluation metrics
   - Updated comparison functions to show RMSE, MAE, R² improvements

3. `predict.py`
   - Updated to predict severity scores
   - Added severity score to label conversion
   - Updated output to show severity scores and interpretations

4. `run_pipeline.py`
   - Updated descriptions to reflect regression

5. `README.md`
   - Updated documentation to explain regression approach
   - Updated performance metrics section
   - Added severity score interpretation guide

## Usage

### Training
```python
from eeg_migraine_classifier import EEGDataProcessor, EEGMigraineRegressor

processor = EEGDataProcessor()
# ... load and clean data ...
X, y = processor.prepare_features(clean_data, labels)
# y is now continuous severity scores [0.0, 0.5, 1.0]

regressor = EEGMigraineRegressor()
results = regressor.train_and_evaluate(X_train, X_test, y_train, y_test)
# Results include RMSE, MAE, R² metrics
```

### Prediction
```python
severity_scores = model.predict(X)
# Returns continuous scores [0.0-1.0]

# Convert to labels if needed
labels = [processor.severity_to_label(score) for score in severity_scores]
```

## Migration Notes

- **Backward Compatibility**: Old classification models will not work with new regression code
- **Data Format**: Same input data format, but labels are now converted to severity scores
- **Model Files**: Need to retrain models (old `.pkl` files are incompatible)
- **Metrics**: Different evaluation metrics require different interpretation

## Example Output

### Regression Output
```
Severity Score: 0.35
Predicted Stage: preictal
Interpretation: -> Preictal (pre-migraine)
```

### Metrics
```
RMSE: 0.1234
MAE: 0.0987
R² Score: 0.8521
```

## Future Enhancements

1. **Custom Thresholds**: Allow users to set custom thresholds for stage classification
2. **Severity Levels**: Add more granular severity levels (e.g., 0.0, 0.25, 0.5, 0.75, 1.0)
3. **Uncertainty Estimation**: Add prediction intervals or confidence scores
4. **Time Series**: Consider temporal patterns for severity progression prediction

