import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, roc_auc_score, confusion_matrix,
    classification_report, cohen_kappa_score, jaccard_score
)
from sklearn.preprocessing import LabelBinarizer

# Global results DataFrame
crdf = pd.DataFrame(columns=[
    'Model', 'Accuracy', 'Precision', 'Recall or Sensitivity',
    'F1 Score', 'Specificity', 'AUC', "Cohen's Kappa", 'Jaccard Similarity'
])

# Optional: customize your class label map
classmap = {
    0: 'Normal',
    1: 'Pneumonia'
}


def run_model(clf, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a classifier.
    """
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
    evaluate_classification(y_test, y_pred, clf.__class__.__name__, y_proba)
    return visualize_results(y_test, y_pred, clf.__class__.__name__), clf


def evaluate_classification(y_true, y_pred, model_name, y_proba=None):
    """
    Compute and store detailed evaluation metrics into global crdf DataFrame.
    """
    global crdf
    is_binary = len(np.unique(y_true)) == 2
    average = 'binary' if is_binary else 'macro'

    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    sp = recall_score(y_true, y_pred, pos_label=0, zero_division=0) if is_binary else np.nan
    roc_auc = roc_auc_score(y_true, y_proba[:, 1]) if (y_proba is not None and is_binary) else np.nan
    cohen_kappa = cohen_kappa_score(y_true, y_pred)
    jaccard_sim = jaccard_score(y_true, y_pred, average=average, zero_division=0)

    crdf = pd.concat([crdf, pd.DataFrame({
        'Model': [model_name], 'Accuracy': [acc], 'Precision': [pre], 'Recall or Sensitivity': [rec],
        'F1 Score': [f1], 'Specificity': [sp], 'AUC': [roc_auc],
        "Cohen's Kappa": [cohen_kappa], 'Jaccard Similarity': [jaccard_sim]
    })], ignore_index=True)


def multiclass_roc_auc_score(y_test, y_pred, model_name):
    """
    Plot and return ROC AUC scores for each class in multiclass problems.
    """
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    y_pred_bin = lb.transform(y_pred)

    roc_auc_values = {}
    plt.figure(figsize=(8, 6))
    for idx, label in enumerate(lb.classes_):
        fpr, tpr, _ = roc_curve(y_test_bin[:, idx], y_pred_bin[:, idx])
        roc_auc = auc(fpr, tpr)
        roc_auc_values[label] = roc_auc
        plt.plot(fpr, tpr, label=f'{classmap.get(label, label)} (AUC: {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig(f'ROC_AUC_{model_name}.jpeg')
    plt.show()
    return roc_auc_values


def visualize_results(y_true, y_pred, model_name):
    """
    Display confusion matrix and classification report.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)

    print('\n=== Classification Report ===')
    print(classification_report(
        y_true, y_pred, digits=4,
        target_names=[classmap.get(label, str(label)) for label in labels]
    ))

    print('Cohen’s Kappa:', cohen_kappa_score(y_true, y_pred))
    print('Jaccard Similarity:', jaccard_score(y_true, y_pred, average='macro'))
    print('ROC AUC Scores:', multiclass_roc_auc_score(y_true, y_pred, model_name))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='g',
        xticklabels=[classmap.get(label, str(label)) for label in labels],
        yticklabels=[classmap.get(label, str(label)) for label in labels],
        cmap="Blues"
    )
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'CM_{model_name}.jpg')
    plt.show()
    return cm


def batch_run(models, X_train, y_train, X_test, y_test):
    """
    Run multiple models sequentially and evaluate each.
    """
    for clf in tqdm(models, desc="Running models"):
        print(f"\n--- Evaluating {clf.__class__.__name__} ---")
        run_model(clf, X_train, y_train, X_test, y_test)
    return crdf
