"""
Complete EEG Migraine Severity Regression Pipeline
1. Load and clean data
2. Train initial regression model
3. Generate synthetic data using VAE/GAN
4. Retrain model on augmented data
5. Evaluate and compare results
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from eeg_migraine_classifier import EEGDataProcessor, EEGMigraineRegressor
from generative_data_augmentation import augment_dataset, EEGDataAugmenter

class CompletePipeline:
    """Complete pipeline for EEG migraine severity regression with data augmentation"""
    
    def __init__(self):
        self.processor = EEGDataProcessor()
        self.regressor = EEGMigraineRegressor()
        self.initial_results = None
        self.augmented_results = None
        
    def load_and_clean_data(self, ictal_path, preictal_path):
        """Load and clean the raw EEG data"""
        print("\n" + "="*70)
        print("STEP 1: LOADING AND CLEANING DATA")
        print("="*70)
        
        # Load data
        ictal_df, preictal_df = self.processor.load_data(ictal_path, preictal_path)
        
        # Clean data
        clean_data, labels, channels = self.processor.clean_data(ictal_df, preictal_df)
        
        return clean_data, labels, channels
    
    def train_initial_model(self, clean_data, labels):
        """Train initial regression model on original (small) dataset"""
        print("\n" + "="*70)
        print("STEP 2: TRAINING INITIAL REGRESSION MODEL (Original Data)")
        print("="*70)
        
        # Prepare features
        X, y = self.processor.prepare_features(clean_data, labels)
        
        # Split data (no stratification for regression)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Severity range: [{y.min():.2f}, {y.max():.2f}]")
        
        # Train models
        results = self.regressor.train_and_evaluate(
            X_train, X_test, y_train, y_test
        )
        
        self.initial_results = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'results': results,
            'rmse': results[self.regressor.best_model_name]['rmse'],
            'mae': results[self.regressor.best_model_name]['mae'],
            'r2': results[self.regressor.best_model_name]['r2']
        }
        
        return results
    
    def augment_data(self, clean_data, labels, target_samples=200, method='vae', epochs=150):
        """Generate synthetic data using VAE or GAN"""
        print("\n" + "="*70)
        print("STEP 3: GENERATING SYNTHETIC DATA")
        print("="*70)
        
        # Separate data by severity level (for augmentation purposes)
        # Convert labels to severity scores for grouping
        severity_scores = np.array([self.processor.severity_map[label] for label in labels])
        class_data = {}
        
        # Group by severity levels: 0.0 (nonictal), 0.5 (preictal), 1.0 (ictal)
        for severity, label_name in self.processor.severity_map.items():
            indices = np.where(severity_scores == severity)[0]
            if len(indices) > 0:
                class_data[label_name] = clean_data.iloc[indices].values
        
        print(f"\nOriginal class distribution:")
        for class_name, data in class_data.items():
            print(f"  {class_name}: {len(data)} samples")
        
        # Augment data
        print(f"\nTarget samples per class: {target_samples}")
        print(f"Augmentation method: {method.upper()}")
        print(f"Training epochs: {epochs}")
        
        augmented_dict = augment_dataset(
            class_data, 
            target_samples_per_class=target_samples,
            method=method,
            epochs=epochs,
            verbose=1
        )
        
        # Combine augmented data
        augmented_data_list = []
        augmented_labels_list = []
        
        for class_name, augmented_class_data in augmented_dict.items():
            n_samples = len(augmented_class_data)
            # Convert to numpy if needed
            if isinstance(augmented_class_data, pd.DataFrame):
                augmented_class_data = augmented_class_data.values
            augmented_data_list.append(augmented_class_data)
            augmented_labels_list.extend([class_name] * n_samples)
            print(f"  {class_name}: {n_samples} samples (augmented)")
        
        # Combine numpy arrays and convert to DataFrame
        combined_array = np.vstack(augmented_data_list)
        augmented_data = pd.DataFrame(combined_array, columns=clean_data.columns)
        augmented_labels = augmented_labels_list
        
        return augmented_data, augmented_labels
    
    def train_augmented_model(self, augmented_data, augmented_labels):
        """Train regression model on augmented dataset"""
        print("\n" + "="*70)
        print("STEP 4: TRAINING REGRESSION MODEL ON AUGMENTED DATA")
        print("="*70)
        
        # Prepare features
        X_aug, y_aug = self.processor.prepare_features(augmented_data, augmented_labels)
        
        # Split data (no stratification for regression)
        X_train_aug, X_test_aug, y_train_aug, y_test_aug = train_test_split(
            X_aug, y_aug, test_size=0.3, random_state=42
        )
        
        print(f"\nTraining set: {X_train_aug.shape[0]} samples")
        print(f"Test set: {X_test_aug.shape[0]} samples")
        print(f"Severity range: [{y_aug.min():.2f}, {y_aug.max():.2f}]")
        
        # Create new regressor for augmented data
        augmented_regressor = EEGMigraineRegressor()
        results_aug = augmented_regressor.train_and_evaluate(
            X_train_aug, X_test_aug, y_train_aug, y_test_aug
        )
        
        self.augmented_results = {
            'X_train': X_train_aug,
            'X_test': X_test_aug,
            'y_train': y_train_aug,
            'y_test': y_test_aug,
            'results': results_aug,
            'rmse': results_aug[augmented_regressor.best_model_name]['rmse'],
            'mae': results_aug[augmented_regressor.best_model_name]['mae'],
            'r2': results_aug[augmented_regressor.best_model_name]['r2'],
            'regressor': augmented_regressor
        }
        
        return results_aug, augmented_regressor
    
    def compare_results(self):
        """Compare initial and augmented regression model results"""
        print("\n" + "="*70)
        print("STEP 5: COMPARING RESULTS")
        print("="*70)
        
        if self.initial_results is None or self.augmented_results is None:
            print("Error: Both initial and augmented results are required!")
            return
        
        initial_rmse = self.initial_results['rmse']
        augmented_rmse = self.augmented_results['rmse']
        initial_mae = self.initial_results['mae']
        augmented_mae = self.augmented_results['mae']
        initial_r2 = self.initial_results['r2']
        augmented_r2 = self.augmented_results['r2']
        
        rmse_improvement = ((initial_rmse - augmented_rmse) / initial_rmse) * 100
        mae_improvement = ((initial_mae - augmented_mae) / initial_mae) * 100
        r2_improvement = ((augmented_r2 - initial_r2) / abs(initial_r2)) * 100 if initial_r2 != 0 else 0
        
        print(f"\nInitial Model:")
        print(f"  RMSE: {initial_rmse:.4f}")
        print(f"  MAE: {initial_mae:.4f}")
        print(f"  R² Score: {initial_r2:.4f}")
        
        print(f"\nAugmented Model:")
        print(f"  RMSE: {augmented_rmse:.4f}")
        print(f"  MAE: {augmented_mae:.4f}")
        print(f"  R² Score: {augmented_r2:.4f}")
        
        print(f"\nImprovement:")
        print(f"  RMSE: {rmse_improvement:+.2f}% (lower is better)")
        print(f"  MAE: {mae_improvement:+.2f}% (lower is better)")
        print(f"  R²: {r2_improvement:+.2f}% (higher is better)")
        
        # Detailed comparison
        print(f"\n{'='*70}")
        print("DETAILED COMPARISON")
        print(f"{'='*70}")
        
        initial_best = self.regressor.best_model_name
        augmented_best = self.augmented_results['regressor'].best_model_name
        
        print(f"\nInitial Best Model: {initial_best}")
        print(f"Augmented Best Model: {augmented_best}")
        
        # Regression metrics comparison
        y_test_initial = self.initial_results['y_test']
        y_pred_initial = self.initial_results['results'][initial_best]['predictions']
        
        y_test_aug = self.augmented_results['y_test']
        y_pred_aug = self.augmented_results['results'][augmented_best]['predictions']
        
        print(f"\n{'='*70}")
        print("INITIAL MODEL - Detailed Metrics")
        print(f"{'='*70}")
        print(f"MSE: {mean_squared_error(y_test_initial, y_pred_initial):.4f}")
        print(f"MAE: {mean_absolute_error(y_test_initial, y_pred_initial):.4f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test_initial, y_pred_initial)):.4f}")
        print(f"R² Score: {r2_score(y_test_initial, y_pred_initial):.4f}")
        print(f"Predicted range: [{y_pred_initial.min():.3f}, {y_pred_initial.max():.3f}]")
        print(f"Actual range: [{y_test_initial.min():.3f}, {y_test_initial.max():.3f}]")
        
        print(f"\n{'='*70}")
        print("AUGMENTED MODEL - Detailed Metrics")
        print(f"{'='*70}")
        print(f"MSE: {mean_squared_error(y_test_aug, y_pred_aug):.4f}")
        print(f"MAE: {mean_absolute_error(y_test_aug, y_pred_aug):.4f}")
        print(f"RMSE: {np.sqrt(mean_squared_error(y_test_aug, y_pred_aug)):.4f}")
        print(f"R² Score: {r2_score(y_test_aug, y_pred_aug):.4f}")
        print(f"Predicted range: [{y_pred_aug.min():.3f}, {y_pred_aug.max():.3f}]")
        print(f"Actual range: [{y_test_aug.min():.3f}, {y_test_aug.max():.3f}]")
        
        print(f"\n{'='*70}")
        print("SUMMARY METRICS")
        print(f"{'='*70}")
        print(f"Initial Model - RMSE: {initial_rmse:.4f}, MAE: {initial_mae:.4f}, R²: {initial_r2:.4f}")
        print(f"Augmented Model - RMSE: {augmented_rmse:.4f}, MAE: {augmented_mae:.4f}, R²: {augmented_r2:.4f}")
        print(f"RMSE Improvement: {rmse_improvement:+.2f}%")
        print(f"MAE Improvement: {mae_improvement:+.2f}%")
        print(f"R² Improvement: {r2_improvement:+.2f}%")
    
    def save_models(self, filepath_prefix='eeg_migraine'):
        """Save trained regression models and processor"""
        print(f"\n{'='*70}")
        print("SAVING MODELS")
        print(f"{'='*70}")
        
        # Save augmented model (best one)
        if self.augmented_results is not None:
            model_path = f'{filepath_prefix}_augmented_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(self.augmented_results['regressor'].best_model, f)
            print(f"Saved augmented regression model to '{model_path}'")
        
        # Save initial model
        if self.initial_results is not None:
            model_path = f'{filepath_prefix}_initial_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(self.regressor.best_model, f)
            print(f"Saved initial regression model to '{model_path}'")
        
        # Save processor
        processor_path = f'{filepath_prefix}_processor.pkl'
        with open(processor_path, 'wb') as f:
            pickle.dump(self.processor, f)
        print(f"Saved processor to '{processor_path}'")
    
    def run_complete_pipeline(self, ictal_path, preictal_path, target_samples=200, 
                             augmentation_method='vae', augmentation_epochs=150):
        """Run the complete regression pipeline"""
        print("\n" + "█"*70)
        print(" " * 15 + "EEG MIGRAINE SEVERITY REGRESSION PIPELINE")
        print("█"*70)
        
        # Step 1: Load and clean
        clean_data, labels, channels = self.load_and_clean_data(ictal_path, preictal_path)
        
        # Step 2: Train initial model
        initial_results = self.train_initial_model(clean_data, labels)
        
        # Step 3: Augment data
        augmented_data, augmented_labels = self.augment_data(
            clean_data, labels, 
            target_samples=target_samples,
            method=augmentation_method,
            epochs=augmentation_epochs
        )
        
        # Step 4: Train augmented model
        augmented_results, augmented_classifier = self.train_augmented_model(
            augmented_data, augmented_labels
        )
        
        # Step 5: Compare results
        self.compare_results()
        
        # Save models
        self.save_models()
        
        print("\n" + "█"*70)
        print(" " * 25 + "PIPELINE COMPLETE!")
        print("█"*70)
        
        return {
            'initial_results': self.initial_results,
            'augmented_results': self.augmented_results,
            'augmented_data': augmented_data,
            'augmented_labels': augmented_labels,
            'channels': channels
        }


def main():
    """Main execution function"""
    # Initialize pipeline
    pipeline = CompletePipeline()
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        ictal_path='raw_ictal.xlsx',
        preictal_path='raw_preictal.xlsx',
        target_samples=200,  # Target samples per severity level
        augmentation_method='vae',  # 'vae' or 'gan'
        augmentation_epochs=150  # Training epochs for generator
    )
    
    return pipeline, results


if __name__ == "__main__":
    pipeline, results = main()

