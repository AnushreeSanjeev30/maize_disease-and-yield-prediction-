# evaluation.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve

def comprehensive_evaluation(model, X_test, y_test, task_name):
    """
    Comprehensive model evaluation with multiple metrics and confusion matrix.
    Note: This was part of the original class but not explicitly called in run_pipeline.
    It's here for completeness.
    """
    print(f"\nComprehensive evaluation for {task_name}:")
    
    y_pred = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f'Confusion Matrix - {task_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_comprehensive_feature_importance(model, feature_names, task_name, top_n=20):
    """
    Plot feature importance with better visualization
    """
    if not hasattr(model, 'feature_importances_'):
        print(f"Model {type(model).__name__} doesn't have feature_importances_ attribute. Skipping plot.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(14, 10)) # Create the figure
    
    # Bar plot
    plt.subplot(2, 1, 1) # Use subplot syntax
    colors = plt.cm.viridis(np.linspace(0, 1, top_n))
    bars = plt.bar(range(top_n), importances[indices], color=colors)
    plt.title(f'Top {top_n} Feature Importances - {task_name}', fontsize=16)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    
    for bar, importance in zip(bars, importances[indices]):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                 f'{importance:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Cumulative importance
    plt.subplot(2, 1, 2) # Use subplot syntax
    cumulative_importance = np.cumsum(importances[indices])
    plt.plot(range(top_n), cumulative_importance, 'ro-', linewidth=2, markersize=6)
    plt.title(f'Cumulative Feature Importance - {task_name}', fontsize=16)
    plt.xlabel('Number of Features', fontsize=12)
    plt.ylabel('Cumulative Importance', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    for i, cum_imp in enumerate(cumulative_importance[::5]):
        plt.annotate(f'{cum_imp:.2f}', (i*5, cum_imp), 
                     textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.show()

# --- ADDED METHOD ---

def plot_learning_curves(self, model, X, y, task_name):
    """
    Plot learning curves to diagnose overfitting
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1_weighted'
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label='Training Score')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', label='Validation Score')
    plt.fill_between(train_sizes, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                     np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
    plt.fill_between(train_sizes, np.mean(val_scores, axis=1) - np.std(val_scores, axis=1),
                     np.mean(val_scores, axis=1) + np.std(val_scores, axis=1), alpha=0.1)
    plt.xlabel('Training Set Size')
    plt.ylabel('F1 Score')
    plt.title(f'Learning Curves - {task_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()