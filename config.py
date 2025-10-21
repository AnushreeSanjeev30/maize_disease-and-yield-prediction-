# config.py
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Enhanced phenotype features with more comprehensive patterns
PHENOTYPE_PATTERNS = [
    'EarWeight', 'KernelWeight', 'EarRowNumber', 'EarLength', 'EarDiameter',
    'TotalKernelVolume', 'SeedSetLength', 'KernelFillPercentage',
    'NorthernLeafBlight', 'SouthernLeafBlight', 'Blight', 'Disease',
    'PlantHeight', 'StalkDiameter', 'TasselBranchNumber', 'SilkingDate',
    'AnthesisDate', 'MaturityDate', 'MoisturePCT', 'TestWeight'
]

# Enhanced envirotype patterns
ENVIROTYPE_PATTERNS = [
    'YEAR_', 'DOY_', 'T2M_MAX_', 'T2M_MIN_', 'T2M_', 'QV2M_', 'RH2M_', 
    'PRECTOTCORR_', 'WS2M_', 'GWETTOP_', 'GWETROOT_', 'ALLSKY_SFC_SW_DWN_', 
    'ALLSKY_SFC_SW_DNI_', 'ALLSKY_SFC_UV_INDEX_', 'TEMP_', 'RAIN_', 
    'HUMID_', 'WIND_', 'SOLAR_', 'PRESSURE_'
]

# Engineered features to look for
ENGINEERED_FEATURES = [
    'avg_ear_weight', 'std_ear_weight', 'max_ear_weight',
    'avg_kernel_weight', 'std_kernel_weight', 'estimated_ear_volume',
    'avg_disease_severity', 'max_disease_severity', 'yield_efficiency'
]

def get_base_models():
    """
    Returns a dictionary of base models for training.
    """
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, 
            max_depth=15, 
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42, 
            n_jobs=1
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=150, 
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            random_state=42
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=200, 
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42, 
            n_jobs=1
        ),
        'LogisticRegression': LogisticRegression(
            class_weight='balanced',
            random_state=42, 
            max_iter=2000,
            C=1.0
        ),
        'SVM': SVC(
            class_weight='balanced',
            random_state=42, 
            probability=True,
            C=1.0,
            kernel='rbf'
        )
    }
    return models