"""
EEG Migraine Severity Regression System
Predicts: Continuous migraine severity score (0.0-1.0) from 23-channel EEG data
0.0 = Non-Ictal (baseline/healthy), 0.5 = Preictal (pre-migraine), 1.0 = Ictal (active migraine)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class EEGDataProcessor:
    """Processes and cleans raw EEG data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        # Severity mapping: nonictal=0.0, preictal=0.5, ictal=1.0
        self.severity_map = {'nonictal': 0.0, 'preictal': 0.5, 'ictal': 1.0}
        
    def load_data(self, ictal_path, preictal_path):
        """Load and combine ictal and preictal data"""
        print("Loading EEG data...")
        
        # Load ictal data
        ictal_df = pd.read_excel(ictal_path)
        print(f"Ictal data shape: {ictal_df.shape}")
        
        # Load preictal data
        preictal_df = pd.read_excel(preictal_path)
        print(f"Preictal data shape: {preictal_df.shape}")
        
        return ictal_df, preictal_df
    
    def clean_data(self, ictal_df, preictal_df):
        """Clean and align the EEG data"""
        print("\nCleaning data...")
        
        # Find common channels between datasets
        ictal_channels = set(ictal_df.columns)
        preictal_channels = set(preictal_df.columns)
        common_channels = list(ictal_channels.intersection(preictal_channels))
        
        # Filter out non-numeric columns and problematic columns
        common_channels = [ch for ch in common_channels 
                          if ictal_df[ch].dtype in [np.float64, np.int64] 
                          and preictal_df[ch].dtype in [np.float64, np.int64]]
        
        # Remove channels with too many missing values
        valid_channels = []
        for ch in common_channels:
            ictal_missing = ictal_df[ch].isna().sum() / len(ictal_df)
            preictal_missing = preictal_df[ch].isna().sum() / len(preictal_df)
            if ictal_missing < 0.8 and preictal_missing < 0.8:  # Keep channels with <80% missing
                valid_channels.append(ch)
        
        print(f"Found {len(valid_channels)} valid common channels")
        
        # Extract valid channels
        ictal_clean = ictal_df[valid_channels].copy()
        preictal_clean = preictal_df[valid_channels].copy()
        
        # Handle missing values
        ictal_clean = pd.DataFrame(
            self.imputer.fit_transform(ictal_clean),
            columns=valid_channels
        )
        preictal_clean = pd.DataFrame(
            self.imputer.transform(preictal_clean),
            columns=valid_channels
        )
        
        # Create labels
        ictal_labels = ['ictal'] * len(ictal_clean)
        preictal_labels = ['preictal'] * len(preictal_clean)
        
        # Combine data
        combined_data = pd.concat([ictal_clean, preictal_clean], ignore_index=True)
        combined_labels = ictal_labels + preictal_labels
        
        # Create non-ictal class (baseline/healthy EEG)
        # Use statistical properties of preictal data to generate baseline
        nonictal_data = self._generate_baseline_data(preictal_clean, n_samples=len(ictal_clean) + len(preictal_clean))
        nonictal_labels = ['nonictal'] * len(nonictal_data)
        
        # Combine all data
        final_data = pd.concat([combined_data, nonictal_data], ignore_index=True)
        final_labels = combined_labels + nonictal_labels
        
        print(f"\nFinal dataset shape: {final_data.shape}")
        print(f"Severity distribution:")
        print(f"  Ictal (1.0): {ictal_labels.count('ictal')} samples")
        print(f"  Preictal (0.5): {preictal_labels.count('preictal')} samples")
        print(f"  Non-Ictal (0.0): {nonictal_labels.count('nonictal')} samples")
        
        # Store feature columns for later use
        self.feature_columns = valid_channels
        
        return final_data, final_labels, valid_channels
    
    def _generate_baseline_data(self, reference_data, n_samples):
        """Generate baseline (non-ictal) EEG data based on reference"""
        # Non-ictal EEG typically has lower amplitude and more stable patterns
        if len(reference_data) >= n_samples:
            baseline_data = reference_data.head(n_samples).copy()
        else:
            baseline_data = reference_data.copy()
        
        # Reduce amplitude by 30-50% and add small noise
        while len(baseline_data) < n_samples:
            sample = reference_data.sample(1).iloc[0].copy()
            # Scale down amplitude (baseline is calmer)
            sample = sample * np.random.uniform(0.5, 0.7)
            # Add small Gaussian noise
            noise = np.random.normal(0, np.abs(sample).std() * 0.1, len(sample))
            sample = sample + noise
            baseline_data = pd.concat([baseline_data, sample.to_frame().T], ignore_index=True)
        
        return baseline_data.head(n_samples)
    
    def prepare_features(self, data, labels):
        """Prepare features for machine learning (regression)"""
        X = data.values
        
        # Convert labels to continuous severity scores
        y = np.array([self.severity_map[label] for label in labels])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def severity_to_label(self, severity_score):
        """Convert severity score back to label name"""
        if severity_score < 0.25:
            return 'nonictal'
        elif severity_score < 0.75:
            return 'preictal'
        else:
            return 'ictal'


class EEGMigraineRegressor:
    """Machine Learning models for EEG migraine severity regression"""
    
    def __init__(self):
        self.models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
            'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        self.best_model = None
        self.best_model_name = None
        
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """Train multiple regression models and evaluate them"""
        print("\n" + "="*60)
        print("TRAINING REGRESSION MODELS")
        print("="*60)
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate regression metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse,
                'predictions': y_pred
            }
            
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R² Score: {r2:.4f}")
            
            # Show prediction range
            print(f"  Predicted range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
            print(f"  Actual range: [{y_test.min():.3f}, {y_test.max():.3f}]")
        
        # Select best model (lowest RMSE)
        best_name = min(results, key=lambda x: results[x]['rmse'])
        self.best_model = results[best_name]['model']
        self.best_model_name = best_name
        
        print(f"\n{'='*60}")
        print(f"Best Model: {best_name}")
        print(f"  RMSE: {results[best_name]['rmse']:.4f}")
        print(f"  MAE: {results[best_name]['mae']:.4f}")
        print(f"  R² Score: {results[best_name]['r2']:.4f}")
        print(f"{'='*60}")
        
        return results
    
    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("Model not trained yet!")
        return self.best_model.predict(X)


def main():
    """Main execution function"""
    print("="*60)
    print("EEG MIGRAINE SEVERITY REGRESSION SYSTEM")
    print("="*60)
    
    # Initialize processor
    processor = EEGDataProcessor()
    
    # Load data
    ictal_df, preictal_df = processor.load_data('raw_ictal.xlsx', 'raw_preictal.xlsx')
    
    # Clean data
    clean_data, labels, channels = processor.clean_data(ictal_df, preictal_df)
    
    # Prepare features
    X, y = processor.prepare_features(clean_data, labels)
    
    # Split data (no stratification for regression)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Severity range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Train models
    regressor = EEGMigraineRegressor()
    results = regressor.train_and_evaluate(
        X_train, X_test, y_train, y_test
    )
    
    # Save model and processor
    import pickle
    
    model_path = 'eeg_regressor_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(regressor.best_model, f)
    
    processor_path = 'eeg_processor.pkl'
    with open(processor_path, 'wb') as f:
        pickle.dump(processor, f)
    
    print(f"\nModel saved to '{model_path}'")
    print(f"Processor saved to '{processor_path}'")
    
    return processor, regressor, clean_data, labels, channels


if __name__ == "__main__":
    processor, regressor, data, labels, channels = main()

