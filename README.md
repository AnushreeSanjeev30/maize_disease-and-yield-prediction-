# Maize Yield and Disease Resistance Prediction

##  Project Overview

This project implements a comprehensive machine learning pipeline for maize (corn) classification based on multi-omics data. The system predicts two critical agricultural traits:

1. **Yield Classification**: Categorizes maize varieties into LOW, MEDIUM, or HIGH yield classes
2. **Disease Resistance**: Classifies varieties as RESISTANT or SUSCEPTIBLE to diseases

The pipeline integrates three types of biological data:
- **Genotype data**: SNP (Single Nucleotide Polymorphism) markers
- **Phenotype data**: Observable plant characteristics (ear weight, plant height, etc.)
- **Envirotype data**: Environmental conditions (temperature, humidity, solar radiation, etc.)

## Key Features

- **Multi-omics data integration**: Combines genetic, phenotypic, and environmental data
- **Intelligent feature engineering**: Creates biologically meaningful composite features
- **Advanced preprocessing**: Handles missing values, scaling, and feature selection
- **Ensemble learning**: Trains multiple ML models and creates voting ensembles
- **Comprehensive evaluation**: Provides detailed performance metrics and visualizations
- **Model persistence**: Saves trained models for future predictions

## Dataset Requirements

### Expected Input Format
- **File type**: CSV format
- **File name**: `newgpe_augmented.csv` (configurable)

### Feature Categories

#### 1. Phenotype Features
Observable plant characteristics with patterns like:
- `EarWeight`, `KernelWeight`, `EarRowNumber`
- `EarLength`, `EarDiameter`, `TotalKernelVolume`
- `NorthernLeafBlight`, `SouthernLeafBlight`
- `PlantHeight`, `StalkDiameter`, `TasselBranchNumber`
- `SilkingDate`, `AnthesisDate`, `MaturityDate`
- `MoisturePCT`, `TestWeight`

#### 2. Envirotype Features
Environmental data with patterns like:
- `YEAR_`, `DOY_` (Day of Year)
- `T2M_MAX_`, `T2M_MIN_`, `T2M_` (Temperature)
- `QV2M_`, `RH2M_` (Humidity)
- `PRECTOTCORR_` (Precipitation)
- `WS2M_` (Wind Speed)
- `GWETTOP_`, `GWETROOT_` (Soil Moisture)
- `ALLSKY_SFC_SW_DWN_`, `ALLSKY_SFC_SW_DNI_` (Solar Radiation)
- `ALLSKY_SFC_UV_INDEX_` (UV Index)

#### 3. Genotype Features
SNP markers in various formats:
- `0/0`, `0/1`, `1/1` (standard genotype notation)
- `0/0:12:30,0:30:99` (extended VCF format)
- `0|0`, `0|1`, `1|1` (phased genotypes)

## Architecture

### Class: `ImprovedMaizeClassificationModel`

#### Key Components

##### Initialization
```python
__init__()
```
Initializes storage for models, scalers, imputers, feature selectors, label encoders, PCA transformers, and feature names.

##### Data Loading & Preprocessing
```python
load_and_preprocess_data(file_path)
```
- Loads CSV data
- Checks for duplicates and missing values
- Provides dataset statistics
- Returns pandas DataFrame

##### Feature Type Separation
```python
separate_feature_types(df)
```
- Automatically identifies phenotype, envirotype, and genotype features
- Uses pattern matching for classification
- Returns dictionary with feature lists

##### Feature Engineering
```python
engineer_phenotype_features(df, phenotype_features)
```
Creates composite features:
- `avg_ear_weight`, `std_ear_weight`, `max_ear_weight`
- `avg_kernel_weight`, `std_kernel_weight`
- `estimated_ear_volume` (cylindrical approximation)
- `avg_disease_severity`, `max_disease_severity`
- `yield_efficiency` ratio

##### Target Variable Creation
```python
derive_improved_target_variables(df, feature_types)
```
- **Yield Classification**: Uses composite scoring from multiple yield indicators
  - Normalizes and weights primary vs. secondary indicators
  - Creates LOW (bottom 30%), MEDIUM (30-70%), HIGH (top 30%) classes
- **Disease Resistance**: Aggregates disease severity scores
  - Uses median split for binary classification
  - Lower severity = RESISTANT, higher = SUSCEPTIBLE

##### Feature Preprocessing
```python
improved_feature_preprocessing(df, feature_types, feature_type)
```
- Converts SNP markers to numeric format
- Removes features with >75% missing values
- Imputes missing values (KNN for phenotype/envirotype, mode for genotype)
- Removes low-variance features
- Scales features (RobustScaler for phenotype/envirotype, StandardScaler for genotype)

##### SNP Conversion
```python
improved_snp_conversion(series)
```
Handles multiple SNP formats:
- `0/0` → 0 (homozygous reference)
- `0/1` → 1 (heterozygous)
- `1/1` → 2 (homozygous alternate)
- Extended formats with quality scores

##### Intelligent Feature Selection
```python
intelligent_feature_selection(X, y, task_name, n_features)
```
- Combines univariate selection (mutual information) with RFE (Recursive Feature Elimination)
- Selects top 500-1000 features depending on dataset size
- Prints detailed feature list for interpretation

##### Model Training
```python
train_ensemble_models(X, y, task_name)
```
Trains five models:
1. **RandomForest**: 200 trees, balanced class weights
2. **GradientBoosting**: 150 estimators, learning rate 0.1
3. **ExtraTrees**: 200 trees, balanced class weights
4. **LogisticRegression**: L2 regularization, balanced weights
5. **SVM**: RBF kernel, probability estimates

Creates **Voting Ensemble** from top 3 models (RF, GB, ET)

##### Evaluation Metrics
```python
comprehensive_evaluation(model, X_test, y_test, task_name)
```
Reports:
- Accuracy
- F1 Score (weighted and macro)
- AUC-ROC (binary or multi-class)
- Classification report (precision, recall, F1 per class)
- Confusion matrix visualization

##### Feature Importance Visualization
```python
plot_comprehensive_feature_importance(model, feature_names, task_name, top_n)
```
- Bar plot of top N important features
- Cumulative importance plot
- Color-coded visualization

##### Model Persistence
```python
save_best_models(results, output_dir)
```
Saves:
- Best yield prediction model
- Best disease prediction model
- Preprocessing objects (scalers, imputers, selectors, encoders)
- Model metadata (scores, timestamps)

```python
load_saved_model(model_path, preprocessing_path)
```
Loads previously saved models and preprocessing pipeline.

## Usage

### Basic Usage

```python
from maize_classification import ImprovedMaizeClassificationModel

# Initialize model
model = ImprovedMaizeClassificationModel()

# Run complete pipeline
results = model.run_improved_pipeline('newgpe_augmented.csv', feature_type='all')

# Save trained models
model.save_best_models(results, output_dir='saved_models')
```

### Feature Type Options

```python
# Use all features (default)
results = model.run_improved_pipeline('data.csv', feature_type='all')

# Use only phenotype features
results = model.run_improved_pipeline('data.csv', feature_type='phenotype')

# Use only genotype features
results = model.run_improved_pipeline('data.csv', feature_type='genotype')

# Use only envirotype features
results = model.run_improved_pipeline('data.csv', feature_type='envirotype')
```

### Loading Saved Models

```python
# Load a previously trained model
model = ImprovedMaizeClassificationModel()
loaded_model, preprocessing = model.load_saved_model(
    'saved_models/best_yield_model_RandomForest_20241021_143052.pkl',
    'saved_models/yield_preprocessing_20241021_143052.pkl'
)

# Use for predictions
new_predictions = loaded_model.predict(new_data)
```

##  Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

**Required versions:**
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.0.0
```

## Pipeline Workflow

```
1. Data Loading
   ↓
2. Feature Type Separation (Genotype, Phenotype, Envirotype)
   ↓
3. Feature Engineering (Composite features)
   ↓
4. Target Variable Creation (Yield & Disease classes)
   ↓
5. Data Preprocessing (Imputation, Scaling, Variance filtering)
   ↓
6. Feature Selection (Intelligent multi-method selection)
   ↓
7. Model Training (5 models + Ensemble)
   ↓
8. Cross-validation (5-fold stratified)
   ↓
9. Model Evaluation (Multiple metrics)
   ↓
10. Visualization (Feature importance, Confusion matrices)
    ↓
11. Model Saving (Best models + preprocessing)
```

##  Output Files

### Saved Models Directory Structure
```
saved_models/
├── best_yield_model_[ModelName]_[Timestamp].pkl
├── yield_preprocessing_[Timestamp].pkl
├── best_disease_model_[ModelName]_[Timestamp].pkl
├── disease_preprocessing_[Timestamp].pkl
└── model_metadata_[Timestamp].pkl
```

### Console Output
- Dataset statistics and feature counts
- Selected features for each task (printed in detail)
- Training progress for each model
- Cross-validation scores (mean ± std)
- Final performance metrics
- Best model selection

### Visualizations
- Confusion matrices for yield and disease predictions
- Feature importance bar plots (top 20 features)
- Cumulative importance curves

##  Performance Metrics

### Yield Prediction (3-class classification)
- **Accuracy**: Overall correctness
- **F1 Score (Weighted)**: Harmonic mean of precision and recall, weighted by class size
- **F1 Score (Macro)**: Unweighted average F1 across classes
- **AUC-ROC (Multi-class)**: One-vs-Rest ROC curve area

### Disease Resistance (Binary classification)
- **Accuracy**: Overall correctness
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **Precision/Recall per class**: Detailed per-class metrics

## Customization

### Adjust Feature Selection
```python
# Modify in intelligent_feature_selection()
n_features = 1000  # Change number of features to select
```

### Modify Model Hyperparameters
```python
# In train_ensemble_models(), adjust model parameters
RandomForestClassifier(
    n_estimators=300,  # Increase trees
    max_depth=20,      # Adjust tree depth
    # ... other parameters
)
```

### Change Classification Thresholds
```python
# In derive_improved_target_variables()
low_threshold = np.percentile(yield_score, 25)   # Bottom 25%
high_threshold = np.percentile(yield_score, 75)  # Top 25%
```

## Biological Interpretation

### Yield Classes
- **LOW**: Bottom 30% performers - may need intervention
- **MEDIUM**: Middle 40% - typical performance
- **HIGH**: Top 30% performers - superior genetics/conditions

### Disease Resistance
- **RESISTANT**: Lower disease severity scores - desirable trait
- **SUSCEPTIBLE**: Higher disease severity - may need protection

### Important Features
Features with high importance scores indicate:
- Strong genetic markers for traits
- Critical environmental conditions
- Key phenotypic predictors

## Contact

For questions or issues:
- Create an issue in the repository
- Contact: [apharnakamathr@gmail.com] , [anushreesawant04@gmail.com]
