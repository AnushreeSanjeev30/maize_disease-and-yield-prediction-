# preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, VarianceThreshold, RFE
)
from sklearn.ensemble import RandomForestClassifier
from config import ENGINEERED_FEATURES

def improved_snp_conversion(series):
    """
    Improved SNP marker conversion with better error handling
    """
    def parse_snp(value):
        if pd.isna(value) or value in ['./.', 'N/A', '', '.']:
            return np.nan
        try:
            if isinstance(value, str):
                if ':' in value:
                    genotype = value.split(':')[0]
                    if '/' in genotype:
                        alleles = genotype.split('/')
                        return int(alleles[0]) + int(alleles[1])
                elif '/' in value:
                    alleles = value.split('/')
                    return int(alleles[0]) + int(alleles[1])
                elif '|' in value:
                    alleles = value.split('|')
                    return int(alleles[0]) + int(alleles[1])
                else:
                    return float(value)
            else:
                return float(value)
        except:
            return np.nan
    
    return series.apply(parse_snp)

def improved_feature_preprocessing(df, feature_types, feature_type='all'):
    """
    Enhanced feature preprocessing.
    Returns X, imputer, scaler, and final feature names.
    """
    print(f"\nPreprocessing {feature_type} features...")
    
    if feature_type == 'all':
        features = feature_types['phenotype'] + feature_types['envirotype'] + feature_types['genotype']
    else:
        features = feature_types[feature_type]
    
    if feature_type in ['phenotype', 'all']:
        features.extend([f for f in ENGINEERED_FEATURES if f in df.columns])
    
    available_features = [col for col in features if col in df.columns]
    X = df[available_features].copy()
    print(f"Starting with {len(available_features)} features")
    
    if feature_type in ['genotype', 'all']:
        genotype_cols = [col for col in available_features if col in feature_types['genotype']]
        for col in genotype_cols:
            if col in X.columns:
                X[col] = improved_snp_conversion(X[col])
    
    missing_threshold = 0.75
    missing_ratio = X.isnull().sum() / len(X)
    features_to_keep = missing_ratio[missing_ratio < missing_threshold].index.tolist()
    X = X[features_to_keep]
    print(f"Features after removing high missing values: {len(features_to_keep)}")
    
    if feature_type == 'genotype':
        imputer = SimpleImputer(strategy='most_frequent')
    else:
        imputer = KNNImputer(n_neighbors=5)
    
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    
    variance_threshold = 0.001 if feature_type == 'genotype' else 0.01
    variance_selector = VarianceThreshold(threshold=variance_threshold)
    X_variance = variance_selector.fit_transform(X)
    selected_features = X.columns[variance_selector.get_support()]
    X = X[selected_features]
    print(f"Features after variance threshold: {len(selected_features)}")
    
    scaler = StandardScaler() if feature_type == 'genotype' else RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X, imputer, scaler, X.columns.tolist()

def intelligent_feature_selection(X, y, task_name, n_features=None):
    """
    Intelligent feature selection using multiple methods.
    Returns selected dataframe and the selector object.
    """
    print(f"\nIntelligent feature selection for {task_name}...")
    
    if n_features is None:
        n_features = min(X.shape[1], 1000) if X.shape[1] > 1000 else min(X.shape[1], 500)
    
    print(f"Selecting top {n_features} features from {X.shape[1]} available features")
    
    # Method 1: Univariate
    univariate_selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    univariate_selector.fit(X, y)
    
    # Method 2: RFE
    rf_selector = RFE(
        estimator=RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1),
        n_features_to_select=n_features,
        step=0.1
    )
    rf_selector.fit(X, y)
    
    univariate_features = set(X.columns[univariate_selector.get_support()])
    rfe_features = set(X.columns[rf_selector.get_support()])
    
    common_features = univariate_features.intersection(rfe_features)
    if len(common_features) < n_features // 2:
        selected_features = list(univariate_features.union(rfe_features))[:n_features]
    else:
        remaining_univariate = univariate_features - common_features
        selected_features = list(common_features) + list(remaining_univariate)[:n_features - len(common_features)]
    
    X_selected = X[selected_features]
    
    # Create a final selector based on the chosen features for transformation
    final_selector = SelectKBest(k=len(selected_features))
    final_selector.fit(X[selected_features], y)
    # Hack to store the selected feature names
    final_selector.selected_features_ = selected_features
    
    print(f"Selected {len(selected_features)} features using intelligent selection")
    print("\nSelected features (top 20):")
    for i, feature in enumerate(selected_features[:20], 1):
        print(f"{i:4d}. {feature}")

    return X_selected, final_selector