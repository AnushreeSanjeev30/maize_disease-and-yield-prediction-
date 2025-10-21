# modeling.py
import numpy as np
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    StratifiedKFold, StratifiedShuffleSplit
)
from sklearn.ensemble import (
    VotingClassifier, RandomForestClassifier, 
    GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from config import get_base_models

def train_ensemble_models(X, y, task_name):
    """
    Train ensemble models with improved hyperparameters.
    Returns results dict and best model name.
    """
    print(f"\nTraining ensemble models for {task_name}...")
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(X, y))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    models = get_base_models()
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        cv_score = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted', n_jobs=1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        if len(np.unique(y)) == 2:
            auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            
        results[name] = {
            'model': model, 'accuracy': accuracy, 'f1_score': f1, 'auc': auc,
            'cv_mean': cv_score.mean(), 'cv_std': cv_score.std(),
            'predictions': y_pred, 'test_labels': y_test
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        print(f"         CV F1: {cv_score.mean():.4f} (+/- {cv_score.std()*2:.4f})")

    # Create ensemble model
    ensemble = VotingClassifier(
        estimators=[
            ('rf', models['RandomForest']),
            ('gb', models['GradientBoosting']),
            ('et', models['ExtraTrees'])
        ],
        voting='soft'
    )
    
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_pred_proba = ensemble.predict_proba(X_test)
    
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')
    
    if len(np.unique(y)) == 2:
        ensemble_auc = roc_auc_score(y_test, ensemble_pred_proba[:, 1])
    else:
        ensemble_auc = roc_auc_score(y_test, ensemble_pred_proba, multi_class='ovr')
            
    results['Ensemble'] = {
        'model': ensemble, 'accuracy': ensemble_accuracy, 'f1_score': ensemble_f1, 
        'auc': ensemble_auc, 'predictions': ensemble_pred, 'test_labels': y_test
    }
    
    print(f"Ensemble - Accuracy: {ensemble_accuracy:.4f}, F1: {ensemble_f1:.4f}, AUC: {ensemble_auc:.4f}")
    
    best_model = max(results.keys(), key=lambda x: results[x]['f1_score'])
    print(f"\nBest model for {task_name}: {best_model} (F1: {results[best_model]['f1_score']:.4f})")
    
    return results, best_model

# --- ADDED METHODS ---

def improved_model_validation(self, X, y, task_name):
    """
    Improved model validation with proper cross-validation
    """
    print(f"\nAdvanced model validation for {task_name}...")
    
    # Use stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define models with improved hyperparameters
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': GradientBoostingClassifier( # Note: original was named XGBoost but used GBC
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            random_state=42
        ),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    
    # Perform cross-validation
    cv_results = {}
    for name, model in models.items():
        print(f"Cross-validating {name}...")
        
        # Multiple scoring metrics
        scoring = ['accuracy', 'f1_weighted', 'roc_auc_ovr']
        
        scores = {}
        for score in scoring:
            cv_score = cross_val_score(model, X, y, cv=skf, scoring=score)
            scores[score] = {
                'mean': cv_score.mean(),
                'std': cv_score.std(),
                'scores': cv_score
            }
        
        cv_results[name] = scores
        
        print(f"  Accuracy: {scores['accuracy']['mean']:.4f} (+/- {scores['accuracy']['std']*2:.4f})")
        print(f"  F1 Score: {scores['f1_weighted']['mean']:.4f} (+/- {scores['f1_weighted']['std']*2:.4f})")
        print(f"  ROC AUC: {scores['roc_auc_ovr']['mean']:.4f} (+/- {scores['roc_auc_ovr']['std']*2:.4f})")
    
    return cv_results

def hyperparameter_tuning(self, X, y, task_name, model_type='RandomForest'):
    """
    Perform hyperparameter tuning with GridSearch
    """
    print(f"\nHyperparameter tuning for {task_name} using {model_type}...")
    
    if model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', None]
        }
    elif model_type == 'GradientBoosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 6, 9],
            'subsample': [0.8, 0.9, 1.0]
        }
    
    # Perform grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='f1_weighted',
        n_jobs=1, verbose=1
    )
    
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_