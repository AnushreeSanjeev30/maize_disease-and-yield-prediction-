# pipeline.py
import pandas as pd
import warnings
import joblib
from sklearn.preprocessing import LabelEncoder

# Import all our custom modules
from data_utils import load_and_preprocess_data, separate_feature_types
from feature_engineering import (
    derive_improved_target_variables,
    advanced_yield_target_creation,  # Added
    advanced_disease_target_creation # Added
)
from preprocessing import (
    improved_feature_preprocessing, 
    intelligent_feature_selection
)
from modeling import (
    train_ensemble_models, 
    improved_model_validation,  # Added
    hyperparameter_tuning     # Added
)
from evaluation import (
    comprehensive_evaluation, 
    plot_comprehensive_feature_importance, 
    plot_learning_curves      # Added
)
from model_io import save_best_models, load_saved_model

warnings.filterwarnings('ignore')

class ImprovedMaizeClassificationModel:
    def __init__(self):
        """
        Initialize the class by creating dictionaries to store state.
        """
        self.models = {}
        self.scalers = {}
        self.imputers = {}
        self.feature_selectors = {}
        self.label_encoders = {}
        self.pca_transformers = {} # Retained from original code
        self.feature_names = {}
    
    def run_improved_pipeline(self, file_path, feature_type='all'):
        """
        Run the improved ML pipeline by calling helper functions.
        This method manages the state of the pipeline.
        """
        print("=" * 80)
        print("IMPROVED MAIZE CLASSIFICATION PIPELINE")
        print("=" * 80)
        
        # 1. Load data
        df = load_and_preprocess_data(file_path)
        if df is None:
            return None
        
        # 2. Separate feature types
        feature_types = separate_feature_types(df)
        
        # 3. Create improved target variables
        df = derive_improved_target_variables(df, feature_types)
        
        # 4. Preprocess features
        # This function returns the state objects, which we store in `self`
        X, imputer, scaler, feature_names = improved_feature_preprocessing(
            df, feature_types, feature_type
        )
        self.imputers[feature_type] = imputer
        self.scalers[feature_type] = scaler
        self.feature_names[feature_type] = feature_names
        
        # 5. Prepare targets
        le_yield = LabelEncoder()
        le_disease = LabelEncoder()
        
        y_yield = le_yield.fit_transform(df['yield_class'])
        y_disease = le_disease.fit_transform(df['disease_resistance'])
        
        # Store label encoders in state
        self.label_encoders['yield'] = le_yield
        self.label_encoders['disease'] = le_disease
        
        print(f"\nFinal feature matrix shape: {X.shape}")
        
        # --- 6. Yield Classification ---
        print("\n" + "="*50)
        print("YIELD CLASSIFICATION")
        print("="*50)
        
        # 6a. Feature Selection
        X_yield, yield_selector = intelligent_feature_selection(
            X, y_yield, 'yield_prediction', n_features=1000
        )
        self.feature_selectors['yield'] = yield_selector
        
        # 6b. Train Models
        yield_results, best_yield_model = train_ensemble_models(
            X_yield, pd.Series(y_yield), 'yield_prediction'
        )
        self.models['yield'] = yield_results
        
        # --- 7. Disease Resistance Classification ---
        print("\n" + "="*50)
        print("DISEASE RESISTANCE CLASSIFICATION")
        print("="*50)
        
        # 7a. Feature Selection
        X_disease, disease_selector = intelligent_feature_selection(
            X, y_disease, 'disease_prediction', n_features=1000
        )
        self.feature_selectors['disease'] = disease_selector
        
        # 7b. Train Models
        disease_results, best_disease_model = train_ensemble_models(
            X_disease, pd.Series(y_disease), 'disease_prediction'
        )
        self.models['disease'] = disease_results
        
        # --- 8. Evaluation & Visualization ---
        print("\n" + "="*50)
        print("COMPREHENSIVE EVALUATION")
        print("="*50)
        
        best_yield_model_obj = yield_results[best_yield_model]['model']
        best_disease_model_obj = disease_results[best_disease_model]['model']
        
        plot_comprehensive_feature_importance(
            best_yield_model_obj, 
            X_yield.columns.tolist(), 
            f'Yield Prediction - {best_yield_model}'
        )
        
        plot_comprehensive_feature_importance(
            best_disease_model_obj, 
            X_disease.columns.tolist(), 
            f'Disease Prediction - {best_disease_model}'
        )
        
        # --- 9. Final Summary ---
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        
        print(f"Best Yield Model: {best_yield_model}")
        print(f"  - Accuracy: {yield_results[best_yield_model]['accuracy']:.4f}")
        print(f"  - F1 Score: {yield_results[best_yield_model]['f1_score']:.4f}")
        print(f"  - AUC: {yield_results[best_yield_model]['auc']:.4f}")
        
        print(f"\nBest Disease Model: {best_disease_model}")
        print(f"  - Accuracy: {disease_results[best_disease_model]['accuracy']:.4f}")
        print(f"  - F1 Score: {disease_results[best_disease_model]['f1_score']:.4f}")
        print(f"  - AUC: {disease_results[best_disease_model]['auc']:.4f}")
        
        # Return all results
        return self.models

    # --- Wrapper methods to call IO functions ---
    
    def save_best_models(self, results, output_dir='saved_models'):
        """
        Wrapper to call the save_best_models function, passing class state.
        """
        return save_best_models(
            results, output_dir, self.scalers, self.imputers,
            self.feature_selectors, self.label_encoders, self.feature_names
        )

    def load_saved_model(self, model_path, preprocessing_path):
        """
        Wrapper to call the load_saved_model function.
        """
        return load_saved_model(model_path, preprocessing_path)

    # --- Other methods from your original class ---
    # These are now available to be called if you need them.
    
    def advanced_yield_target_creation(self, df, feature_types):
        return advanced_yield_target_creation(self, df, feature_types)
    
    def advanced_disease_target_creation(self, df, feature_types):
        return advanced_disease_target_creation(self, df, feature_types)

    def improved_model_validation(self, X, y, task_name):
        return improved_model_validation(self, X, y, task_name)
    
    def hyperparameter_tuning(self, X, y, task_name, model_type='RandomForest'):
        return hyperparameter_tuning(self, X, y, task_name, model_type)

    def plot_learning_curves(self, model, X, y, task_name):
        return plot_learning_curves(self, model, X, y, task_name)
    
    def comprehensive_evaluation(self, model, X_test, y_test, task_name):
        return comprehensive_evaluation(model, X_test, y_test, task_name)