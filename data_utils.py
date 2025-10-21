# data_utils.py
import pandas as pd
from config import PHENOTYPE_PATTERNS, ENVIROTYPE_PATTERNS

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the combined dataset with better error handling
    """
    print("Loading dataset...")
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        
        # Display basic info
        print("\nDataset Info:")
        print(f"Total columns: {len(df.columns)}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Found {duplicates} duplicate rows - removing them")
            df = df.drop_duplicates()
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def separate_feature_types(df):
    """
    Improved feature type separation with better pattern matching
    """
    print("\nSeparating feature types...")
    
    # Identify feature types
    phenotype_features = []
    envirotype_features = []
    
    for col in df.columns:
        if any(pattern in col for pattern in PHENOTYPE_PATTERNS):
            phenotype_features.append(col)
        elif any(pattern in col for pattern in ENVIROTYPE_PATTERNS):
            envirotype_features.append(col)
    
    # Remaining features are genotype (SNP markers)
    genotype_features = [col for col in df.columns 
                         if col not in phenotype_features + envirotype_features]
    
    print(f"Phenotype features: {len(phenotype_features)}")
    print(f"Envirotype features: {len(envirotype_features)}")
    print(f"Genotype features: {len(genotype_features)}")
    
    return {
        'phenotype': phenotype_features,
        'envirotype': envirotype_features,
        'genotype': genotype_features
    }