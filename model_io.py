# model_io.py
import os
import joblib
from datetime import datetime

def save_best_models(results, output_dir, scalers, imputers, feature_selectors, label_encoders, feature_names):
    """
    Save the best models and all corresponding preprocessing objects
    """
    print(f"\nSaving best models to {output_dir}...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = {'timestamp': timestamp}

    # Save yield prediction model
    if 'yield' in results:
        yield_results = results['yield']
        best_yield_model = max(yield_results.keys(), key=lambda x: yield_results[x]['f1_score'])
        best_yield_model_obj = yield_results[best_yield_model]['model']
        
        yield_model_path = os.path.join(output_dir, f'best_yield_model_{best_yield_model}_{timestamp}.pkl')
        joblib.dump(best_yield_model_obj, yield_model_path)
        
        yield_preprocessing_path = os.path.join(output_dir, f'yield_preprocessing_{timestamp}.pkl')
        yield_preprocessing = {
            'scaler': scalers.get('all', None),
            'imputer': imputers.get('all', None),
            'feature_selector': feature_selectors.get('yield', None),
            'label_encoder': label_encoders.get('yield', None),
            'feature_names': feature_names.get('all', None)
        }
        joblib.dump(yield_preprocessing, yield_preprocessing_path)
        
        print(f"✓ Yield model saved: {yield_model_path}")
        print(f"✓ Yield preprocessing saved: {yield_preprocessing_path}")
        
        metadata['yield_model'] = {
            'name': best_yield_model,
            'f1_score': yield_results[best_yield_model]['f1_score'],
            'accuracy': yield_results[best_yield_model]['accuracy'],
            'auc': yield_results[best_yield_model]['auc']
        }

    # Save disease prediction model
    if 'disease' in results:
        disease_results = results['disease']
        best_disease_model = max(disease_results.keys(), key=lambda x: disease_results[x]['f1_score'])
        best_disease_model_obj = disease_results[best_disease_model]['model']
        
        disease_model_path = os.path.join(output_dir, f'best_disease_model_{best_disease_model}_{timestamp}.pkl')
        joblib.dump(best_disease_model_obj, disease_model_path)
        
        disease_preprocessing_path = os.path.join(output_dir, f'disease_preprocessing_{timestamp}.pkl')
        disease_preprocessing = {
            'scaler': scalers.get('all', None),
            'imputer': imputers.get('all', None),
            'feature_selector': feature_selectors.get('disease', None),
            'label_encoder': label_encoders.get('disease', None),
            'feature_names': feature_names.get('all', None)
        }
        joblib.dump(disease_preprocessing, disease_preprocessing_path)
        
        print(f"✓ Disease model saved: {disease_model_path}")
        print(f"✓ Disease preprocessing saved: {disease_preprocessing_path}")
        
        metadata['disease_model'] = {
            'name': best_disease_model,
            'f1_score': disease_results[best_disease_model]['f1_score'],
            'accuracy': disease_results[best_disease_model]['accuracy'],
            'auc': disease_results[best_disease_model]['auc']
        }
    
    metadata_path = os.path.join(output_dir, f'model_metadata_{timestamp}.pkl')
    joblib.dump(metadata, metadata_path)
    print(f"✓ Model metadata saved: {metadata_path}")
    
    print(f"\n🎉 All models and preprocessing objects saved successfully!")
    return output_dir

def load_saved_model(model_path, preprocessing_path):
    """
    Load a saved model and its preprocessing objects
    """
    print(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
        preprocessing = joblib.load(preprocessing_path)
        print("✓ Model and preprocessing objects loaded successfully!")
        return model, preprocessing
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None