import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, Dropout, GlobalAveragePooling2D)
from tensorflow.keras.applications import (
    VGG16, InceptionV3, Xception, DenseNet201, NASNetLarge, ConvNeXtXLarge
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    cohen_kappa_score, jaccard_score, roc_curve, auc
)
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os

# Global result storage
crdf = pd.DataFrame()

class TransferModelTrainer:
    def __init__(self, model_name, model_json_path, model_weights_path):
        self.model_name = model_name
        self.model = self._load_model(model_json_path, model_weights_path)

    def _load_model(self, json_path, weights_path):
        with open(json_path, 'r') as file:
            loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_path)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        return model

    def cross_validate(self, X, Y, class_names, n_splits=10):
        global crdf
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        Y_labels = np.argmax(Y, axis=1)

        fold = 1
        for train_index, test_index in skf.split(X, Y_labels):
            print(f"Running Fold {fold}...")
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            # Retrain model to reset weights
            model = tf.keras.models.clone_model(self.model)
            model.set_weights(self.model.get_weights())
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                          loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'])
            model.fit(X_train, Y_train, epochs=10, batch_size=32, verbose=0)

            y_pred = model.predict(X_test)
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_true_labels = np.argmax(Y_test, axis=1)

            acc = accuracy_score(y_true_labels, y_pred_labels)
            kappa = cohen_kappa_score(y_true_labels, y_pred_labels)
            jaccard = jaccard_score(y_true_labels, y_pred_labels, average='macro')

            report_dict = classification_report(y_true_labels, y_pred_labels, target_names=class_names, output_dict=True)

            row = {
                'Fold': fold,
                'Model': self.model_name,
                'Accuracy': acc,
                "Cohen's Kappa": kappa,
                'Jaccard Similarity': jaccard
            }

            for label in class_names:
                row[f'{label}_Precision'] = report_dict[label]['precision']
                row[f'{label}_Recall'] = report_dict[label]['recall']
                row[f'{label}_F1-Score'] = report_dict[label]['f1-score']

            crdf = pd.concat([crdf, pd.DataFrame([row])], ignore_index=True)
            fold += 1

        crdf.to_csv(f"CrossVal_{self.model_name}.csv", index=False)
        print(f"Cross-validation results saved to CrossVal_{self.model_name}.csv")
        return crdf

