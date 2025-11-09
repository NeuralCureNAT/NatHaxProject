"""
Improved data augmentation with better parameters for higher quality synthetic data
"""

from main_pipeline import CompletePipeline

def main():
    """Run pipeline with improved augmentation settings"""
    print("="*70)
    print("IMPROVED DATA AUGMENTATION PIPELINE")
    print("="*70)
    print("\nUsing optimized parameters for better synthetic data quality...")
    print("This will take longer but should produce better results.\n")
    
    # Initialize pipeline
    pipeline = CompletePipeline()
    
    # Run with improved settings
    results = pipeline.run_complete_pipeline(
        ictal_path='raw_ictal.xlsx',
        preictal_path='raw_preictal.xlsx',
        target_samples=200,  # More samples for better training
        augmentation_method='vae',  # VAE typically works better for small datasets
        augmentation_epochs=150  # More epochs for better quality
    )
    
    print("\n" + "="*70)
    print("Improved augmentation complete!")
    print("="*70)
    print("\nNote: If you want even better results, try:")
    print("  - Increasing target_samples to 300-500")
    print("  - Increasing augmentation_epochs to 200-300")
    print("  - Using 'gan' method for larger datasets")
    
    return pipeline, results

if __name__ == "__main__":
    pipeline, results = main()

