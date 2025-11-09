"""
Quick run script for the EEG Migraine Severity Regression Pipeline
"""

from main_pipeline import CompletePipeline
import sys

def main():
    print("Starting EEG Migraine Severity Regression Pipeline...")
    print("="*70)
    
    # Initialize pipeline
    pipeline = CompletePipeline()
    
    # Run complete pipeline
    # For faster testing: target_samples=50, epochs=30
    # For better quality: target_samples=200, epochs=150
    # For production: target_samples=300-500, epochs=200-300
    results = pipeline.run_complete_pipeline(
        ictal_path='raw_ictal.xlsx',
        preictal_path='raw_preictal.xlsx',
        target_samples=100,  # Target samples per class
        augmentation_method='vae',  # 'vae' or 'gan'
        augmentation_epochs=100  # Training epochs (increase for better quality)
    )
    
    print("\nPipeline completed successfully!")
    return pipeline, results

if __name__ == "__main__":
    try:
        pipeline, results = main()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

