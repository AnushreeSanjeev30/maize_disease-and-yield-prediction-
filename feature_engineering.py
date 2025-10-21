# feature_engineering.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as SK_StandardScaler

def engineer_phenotype_features(df, phenotype_features):
    """
    Create biologically meaningful phenotype features
    """
    print("\nEngineering phenotype features...")
    
    # Create yield-related composite features
    ear_weight_cols = [col for col in phenotype_features if 'EarWeight' in col]
    kernel_weight_cols = [col for col in phenotype_features if 'KernelWeight' in col]
    ear_length_cols = [col for col in phenotype_features if 'EarLength' in col]
    ear_diameter_cols = [col for col in phenotype_features if 'EarDiameter' in col]
    
    if ear_weight_cols:
        df['avg_ear_weight'] = df[ear_weight_cols].mean(axis=1)
        df['std_ear_weight'] = df[ear_weight_cols].std(axis=1)
        df['max_ear_weight'] = df[ear_weight_cols].max(axis=1)
    
    if kernel_weight_cols:
        df['avg_kernel_weight'] = df[kernel_weight_cols].mean(axis=1)
        df['std_kernel_weight'] = df[kernel_weight_cols].std(axis=1)
    
    if ear_length_cols and ear_diameter_cols:
        avg_length = df[ear_length_cols].mean(axis=1)
        avg_diameter = df[ear_diameter_cols].mean(axis=1)
        df['estimated_ear_volume'] = np.pi * (avg_diameter/2)**2 * avg_length
    
    # Disease resistance features
    disease_cols = [col for col in phenotype_features if 'Blight' in col or 'Disease' in col]
    if disease_cols:
        df['avg_disease_severity'] = df[disease_cols].mean(axis=1)
        df['max_disease_severity'] = df[disease_cols].max(axis=1)
    
    # Yield efficiency features
    if 'avg_ear_weight' in df.columns and 'avg_kernel_weight' in df.columns:
        df['yield_efficiency'] = df['avg_ear_weight'] / (df['avg_kernel_weight'] + 1e-6)
    
    return df

def derive_improved_target_variables(df, feature_types):
    """
    Improved target variable derivation with better biological relevance
    """
    print("\nDeriving improved target variables...")
    
    # Engineer phenotype features first
    df = engineer_phenotype_features(df, feature_types['phenotype'])
    
    # --- Improved Yield Classification ---
    yield_indicators = []
    if 'avg_ear_weight' in df.columns: yield_indicators.append('avg_ear_weight')
    if 'avg_kernel_weight' in df.columns: yield_indicators.append('avg_kernel_weight')
    if 'estimated_ear_volume' in df.columns: yield_indicators.append('estimated_ear_volume')
    
    ear_weight_cols = [col for col in df.columns if 'EarWeight' in col]
    kernel_volume_cols = [col for col in df.columns if 'TotalKernelVolume' in col]
    
    all_yield_features = yield_indicators + ear_weight_cols + kernel_volume_cols
    all_yield_features = [col for col in all_yield_features if col in df.columns]
    
    if all_yield_features:
        yield_data = df[all_yield_features].copy()
        for col in yield_data.columns:
            median_val = yield_data[col].median()
            yield_data[col] = yield_data[col].fillna(median_val)
        
        scaler = SK_StandardScaler()
        yield_data_scaled = scaler.fit_transform(yield_data)
        
        if len(yield_indicators) > 0:
            primary_weight = 0.6
            secondary_weight = 0.4
            primary_score = yield_data_scaled[:, :len(yield_indicators)].mean(axis=1)
            secondary_score = yield_data_scaled[:, len(yield_indicators):].mean(axis=1)
            yield_score = primary_weight * primary_score + secondary_weight * secondary_score
        else:
            yield_score = yield_data_scaled.mean(axis=1)
        
        low_threshold = np.percentile(yield_score, 30)
        high_threshold = np.percentile(yield_score, 70)
        
        df['yield_class'] = pd.cut(yield_score, 
                                 bins=[-np.inf, low_threshold, high_threshold, np.inf],
                                 labels=['LOW', 'MEDIUM', 'HIGH'])
    else:
        print("Warning: No yield features found, creating balanced random classes")
        df['yield_class'] = np.random.choice(['LOW', 'MEDIUM', 'HIGH'], size=len(df))
    
    # --- Improved Disease Resistance Classification ---
    disease_features = []
    if 'avg_disease_severity' in df.columns: disease_features.append('avg_disease_severity')
    if 'max_disease_severity' in df.columns: disease_features.append('max_disease_severity')
    blight_cols = [col for col in df.columns if 'Blight' in col]
    disease_features.extend(blight_cols)
    disease_features = [col for col in disease_features if col in df.columns]
    
    if disease_features:
        disease_data = df[disease_features].copy()
        for col in disease_data.columns:
            median_val = disease_data[col].median()
            disease_data[col] = disease_data[col].fillna(median_val)
        
        disease_score = disease_data.mean(axis=1)
        disease_threshold = disease_score.median()
        df['disease_resistance'] = np.where(
            disease_score <= disease_threshold, 'RESISTANT', 'SUSCEPTIBLE'
        )
    else:
        print("Warning: No disease features found, creating balanced random classes")
        df['disease_resistance'] = np.random.choice(['RESISTANT', 'SUSCEPTIBLE'], size=len(df))
    
    print(f"Yield class distribution:\n{df['yield_class'].value_counts(normalize=True)}")
    print(f"Disease resistance distribution:\n{df['disease_resistance'].value_counts(normalize=True)}")
    
    return df

# --- ADDED METHODS ---

def advanced_yield_target_creation(self, df, feature_types):
    """
    Advanced yield target creation using domain knowledge
    """
    print("\nCreating advanced yield targets...")
    
    # Get all potential yield-related features
    yield_features = []
    
    # Primary yield indicators
    primary_yield_patterns = [
        'EarWeight', 'KernelWeight', 'GrainWeight', 'Yield',
        'TotalKernelVolume', 'TestWeight'
    ]
    
    # Secondary yield indicators
    secondary_yield_patterns = [
        'EarLength', 'EarDiameter', 'EarRowNumber', 'KernelFill',
        'SeedSetLength', 'KernelNumber'
    ]
    
    # Collect primary yield features
    for col in df.columns:
        if any(pattern in col for pattern in primary_yield_patterns):
            if df[col].dtype in ['int64', 'float64']:
                yield_features.append(col)
    
    # Collect secondary yield features
    secondary_features = []
    for col in df.columns:
        if any(pattern in col for pattern in secondary_yield_patterns):
            if df[col].dtype in ['int64', 'float64']:
                secondary_features.append(col)
    
    print(f"Found {len(yield_features)} primary yield features")
    print(f"Found {len(secondary_features)} secondary yield features")
    
    if len(yield_features) == 0 and len(secondary_features) == 0:
        print("Warning: No yield-related features found, using random classification")
        return np.random.choice(['LOW', 'MEDIUM', 'HIGH'], size=len(df))
    
    # Create composite yield score
    all_yield_features = yield_features + secondary_features
    yield_data = df[all_yield_features].copy()
    
    # Handle missing values with median imputation
    for col in yield_data.columns:
        if yield_data[col].isnull().sum() > 0:
            median_val = yield_data[col].median()
            yield_data[col] = yield_data[col].fillna(median_val)
    
    # Weight primary features more heavily
    if len(yield_features) > 0:
        primary_data = yield_data[yield_features]
        primary_score = primary_data.mean(axis=1)
        primary_weight = 0.7
    else:
        primary_score = 0
        primary_weight = 0
    
    if len(secondary_features) > 0:
        secondary_data = yield_data[secondary_features]
        secondary_score = secondary_data.mean(axis=1)
        secondary_weight = 1.0 - primary_weight
    else:
        secondary_score = 0
        secondary_weight = 0
    
    # Combine scores
    if primary_weight > 0 and secondary_weight > 0:
        # Normalize scores before combining
        if primary_score.std() > 0:
            primary_score = (primary_score - primary_score.mean()) / primary_score.std()
        if secondary_score.std() > 0:
            secondary_score = (secondary_score - secondary_score.mean()) / secondary_score.std()
        
        composite_score = primary_weight * primary_score + secondary_weight * secondary_score
    elif primary_weight > 0:
        composite_score = primary_score
    else:
        composite_score = secondary_score
    
    # Create classes using more sophisticated approach
    q25 = np.percentile(composite_score, 25)
    q75 = np.percentile(composite_score, 75)
    
    yield_classes = pd.cut(composite_score, 
                           bins=[-np.inf, q25, q75, np.inf],
                           labels=['LOW', 'MEDIUM', 'HIGH'])
    
    return yield_classes

def advanced_disease_target_creation(self, df, feature_types):
    """
    Advanced disease resistance target creation
    """
    print("\nCreating advanced disease resistance targets...")
    
    # Get all disease-related features
    disease_patterns = [
        'Blight', 'Disease', 'Rust', 'Infection', 'Pathogen',
        'Resistance', 'Susceptibility', 'Severity'
    ]
    
    disease_features = []
    for col in df.columns:
        if any(pattern in col for pattern in disease_patterns):
            if df[col].dtype in ['int64', 'float64']:
                disease_features.append(col)
    
    print(f"Found {len(disease_features)} disease-related features")
    
    if len(disease_features) == 0:
        print("Warning: No disease-related features found, using random classification")
        return np.random.choice(['RESISTANT', 'SUSCEPTIBLE'], size=len(df))
    
    # Create composite disease score
    disease_data = df[disease_features].copy()
    
    # Handle missing values
    for col in disease_data.columns:
        if disease_data[col].isnull().sum() > 0:
            median_val = disease_data[col].median()
            disease_data[col] = disease_data[col].fillna(median_val)
    
    # Calculate average disease severity
    disease_score = disease_data.mean(axis=1)
    
    # Use median split
    threshold = disease_score.median()
    
    # Lower disease score = more resistant
    disease_classes = np.where(disease_score <= threshold, 'RESISTANT', 'SUSCEPTIBLE')
    
    return disease_classes