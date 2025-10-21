# main.py
import traceback
from pipeline import ImprovedMaizeClassificationModel

def main():
    """
    Main execution function
    """
    print("Initializing Improved Maize Classification Model...")
    
    # Initialize the model
    model = ImprovedMaizeClassificationModel()
    
    # Define the data file path
    file_path = 'newgpe.csv'
    
    try:
        # Run the full pipeline using 'all' features
        results = model.run_improved_pipeline(file_path, feature_type='all')
        
        if results:
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            
            # Save the best models and preprocessing objects
            model.save_best_models(results, output_dir='saved_models')
            
        else:
            print("Pipeline failed to complete.")
            
    except Exception as e:
        print(f"An error occurred during the pipeline execution: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()