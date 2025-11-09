"""
Prediction script for EEG Migraine Severity Regression
Predicts continuous severity score (0.0-1.0)
0.0 = Non-Ictal (baseline), 0.5 = Preictal (pre-migraine), 1.0 = Ictal (active migraine)
"""

import pickle
import pandas as pd
import numpy as np
import sys

def load_model_and_processor():
    """Load trained regression model and processor"""
    try:
        with open('eeg_migraine_augmented_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('eeg_migraine_processor.pkl', 'rb') as f:
            processor = pickle.load(f)
        
        return model, processor
    except FileNotFoundError as e:
        print(f"Error: Model files not found. Please run the pipeline first.")
        print(f"Missing file: {e.filename}")
        sys.exit(1)

def predict_eeg_data(model, processor, eeg_data):
    """
    Predict migraine severity from EEG data
    
    Args:
        model: Trained regression model
        processor: EEGDataProcessor instance
        eeg_data: DataFrame or numpy array with EEG data (same channels as training data)
    
    Returns:
        severity_scores: Array of predicted severity scores (0.0-1.0)
        predicted_labels: Array of predicted stage labels (nonictal/preictal/ictal)
    """
    # Ensure data is DataFrame
    if isinstance(eeg_data, np.ndarray):
        eeg_data = pd.DataFrame(eeg_data)
    
    # Get feature columns
    if hasattr(processor, 'feature_columns'):
        eeg_data = eeg_data[processor.feature_columns]
    
    # Prepare features using the same preprocessing
    X = eeg_data.values
    
    # Handle missing values
    X_imputed = processor.imputer.transform(X)
    
    # Scale features
    X_scaled = processor.scaler.transform(X_imputed)
    
    # Make predictions (continuous severity scores)
    severity_scores = model.predict(X_scaled)
    
    # Clip predictions to valid range [0.0, 1.0]
    severity_scores = np.clip(severity_scores, 0.0, 1.0)
    
    # Convert severity scores to labels
    predicted_labels = [processor.severity_to_label(score) for score in severity_scores]
    
    return severity_scores, predicted_labels

def predict_from_file(model, processor, file_path, sheet_name=0):
    """Predict from Excel file"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Loaded data from {file_path}")
        print(f"Data shape: {df.shape}")
        
        severity_scores, predicted_labels = predict_eeg_data(model, processor, df)
        
        return severity_scores, predicted_labels, df
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def main():
    """Main prediction function"""
    print("="*70)
    print("EEG MIGRAINE SEVERITY PREDICTION")
    print("="*70)
    print("Severity Score: 0.0 = Non-Ictal, 0.5 = Preictal, 1.0 = Ictal")
    print("="*70)
    
    # Load model and processor
    print("\nLoading model and processor...")
    model, processor = load_model_and_processor()
    print("Model loaded successfully!")
    
    # Check if file path is provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"\nPredicting from file: {file_path}")
        severity_scores, predicted_labels, data = predict_from_file(model, processor, file_path)
        
        # Display results
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        
        for i in range(len(severity_scores)):
            print(f"\nSample {i+1}:")
            print(f"  Severity Score: {severity_scores[i]:.3f}")
            print(f"  Predicted Stage: {predicted_labels[i]}")
            print(f"  Interpretation:")
            if severity_scores[i] < 0.25:
                print(f"    -> Non-Ictal (baseline/healthy)")
            elif severity_scores[i] < 0.75:
                print(f"    -> Preictal (pre-migraine)")
            else:
                print(f"    -> Ictal (active migraine)")
        
        print(f"\n{'='*70}")
        print(f"Total predictions: {len(severity_scores)}")
        print(f"Average severity: {severity_scores.mean():.3f}")
        print(f"Severity range: [{severity_scores.min():.3f}, {severity_scores.max():.3f}]")
        print(f"{'='*70}")
        
        return
    
    # Interactive mode - predict from existing data files
    print("\nNo file provided. Using test data from training files...")
    
    # Load and process test data
    from eeg_migraine_classifier import EEGDataProcessor
    test_processor = EEGDataProcessor()
    ictal_df, preictal_df = test_processor.load_data('raw_ictal.xlsx', 'raw_preictal.xlsx')
    clean_data, labels, channels = test_processor.clean_data(ictal_df, preictal_df)
    
    # Use processor from saved model for consistency
    X, y = processor.prepare_features(clean_data, labels)
    
    # Make predictions
    severity_scores = model.predict(X)
    severity_scores = np.clip(severity_scores, 0.0, 1.0)
    predicted_labels = [processor.severity_to_label(score) for score in severity_scores]
    actual_labels = labels
    actual_severities = y
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    for i in range(min(10, len(predicted_labels))):
        print(f"\nSample {i+1}:")
        print(f"  Actual: {actual_labels[i]} (severity: {actual_severities[i]:.2f})")
        print(f"  Predicted: {predicted_labels[i]} (severity: {severity_scores[i]:.3f})")
        error = abs(severity_scores[i] - actual_severities[i])
        print(f"  Error: {error:.3f}")
    
    # Calculate regression metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(actual_severities, severity_scores)
    mae = mean_absolute_error(actual_severities, severity_scores)
    r2 = r2_score(actual_severities, severity_scores)
    rmse = np.sqrt(mse)
    
    print(f"\n{'='*70}")
    print("MODEL PERFORMANCE METRICS")
    print(f"{'='*70}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

