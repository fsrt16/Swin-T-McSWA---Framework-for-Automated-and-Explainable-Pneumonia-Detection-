import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, cohen_kappa_score,
    jaccard_score, precision_recall_curve, average_precision_score,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import model_from_json


class XAI_Evaluator:
    def __init__(self, model_json_path, model_weights_path, class_names):
        self.model = self.load_model(model_json_path, model_weights_path)
        self.class_names = class_names
        self.encoder = LabelBinarizer()
        self.encoder.fit(class_names)
        self.results = pd.DataFrame()

    def load_model(self, json_path, weights_path):
        with open(json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_path)
        print("Loaded model from disk")
        return model

    def evaluate(self, X, Y, model_name="DL Model"):
        y_pred_probs = self.model.predict(X, verbose=0)
        y_pred = self.encoder.inverse_transform(y_pred_probs)
        y_true = self.encoder.inverse_transform(Y)

        print("\nClassification Report:\n")
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        print(classification_report(y_true, y_pred, target_names=self.class_names, digits=4))

        cm = confusion_matrix(y_true, y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        jaccard = jaccard_score(y_true, y_pred, average='macro')

        result = {
            'Model': model_name,
            'Accuracy': accuracy,
            "Cohen's Kappa": kappa,
            'Jaccard Similarity': jaccard
        }

        for cls in self.class_names:
            result[f'{cls}_Precision'] = report[cls]['precision']
            result[f'{cls}_Recall'] = report[cls]['recall']
            result[f'{cls}_F1-Score'] = report[cls]['f1-score']

        self.results = pd.concat([self.results, pd.DataFrame([result])], ignore_index=True)

        self.plot_confusion_matrix(cm)
        self.plot_pr_curve(y_true, y_pred)
        self.plot_roc_curve(y_true, y_pred, model_name)

        return self.results

    def plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6))
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                         xticklabels=self.class_names, yticklabels=self.class_names,
                         linewidths=0.5, linecolor='black')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                value = cm[i, j]
                color = "white" if value > cm.max() / 2 else "black"
                ax.text(j + 0.5, i + 0.5, f"{value}", ha='center', va='center', color=color, fontsize=12)
        plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
        plt.ylabel('Actual Class', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_pr_curve(self, y_true, y_pred):
        y_true_bin = self.encoder.transform(y_true)
        y_pred_bin = self.encoder.transform(y_pred)

        plt.figure(figsize=(10, 6))
        for i, cls in enumerate(self.class_names):
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_bin[:, i])
            ap = average_precision_score(y_true_bin[:, i], y_pred_bin[:, i])
            plt.plot(recall, precision, label=f'{cls} (AP: {ap:.3f})', linewidth=2)

        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.4, linestyle='--')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.show()

    def plot_roc_curve(self, y_true, y_pred, title):
        y_true_bin = self.encoder.transform(y_true)
        y_pred_bin = self.encoder.transform(y_pred)

        plt.figure(figsize=(10, 6))
        for i, cls in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{cls} (AUC: {roc_auc:.3f})', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {title}', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.4, linestyle='--')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
