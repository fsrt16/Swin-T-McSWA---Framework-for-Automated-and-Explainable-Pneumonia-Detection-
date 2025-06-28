import tensorflow as tf
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
import pandas as pd
import numpy as np

# Global result storage
crdf = pd.DataFrame()

class TransferModelTrainer:
    def __init__(self, model_name, backbone, input_shape=(224, 224, 3), num_classes=3):
        self.model_name = model_name
        self.backbone = backbone
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        base_model = self.backbone(weights='imagenet', include_top=False, input_shape=self.input_shape)
        base_model.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=10):
        print(f"\nTraining {self.model_name}...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        self.history = history
        return history

    def evaluate(self, X_test, y_test, class_names):
        print(f"\nEvaluating {self.model_name}...")
        y_pred = self.model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_true_labels = np.argmax(y_test, axis=1)

        report = classification_report(y_true_labels, y_pred_labels, target_names=class_names, digits=4)
        print(report)

        acc = accuracy_score(y_true_labels, y_pred_labels)
        kappa = cohen_kappa_score(y_true_labels, y_pred_labels)
        jaccard = jaccard_score(y_true_labels, y_pred_labels, average='macro')

        cm = confusion_matrix(y_true_labels, y_pred_labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

        global crdf
        row = {
            'Model': self.model_name,
            'Accuracy': acc,
            "Cohen's Kappa": kappa,
            'Jaccard Similarity': jaccard
        }
        crdf = pd.concat([crdf, pd.DataFrame([row])], ignore_index=True)

        # Per-class metrics
        report_dict = classification_report(y_true_labels, y_pred_labels, target_names=class_names, output_dict=True)
        for label in class_names:
            crdf.loc[crdf.index[-1], f'{label}_Precision'] = report_dict[label]['precision']
            crdf.loc[crdf.index[-1], f'{label}_Recall'] = report_dict[label]['recall']
            crdf.loc[crdf.index[-1], f'{label}_F1-Score'] = report_dict[label]['f1-score']

        return crdf

    def plot_training_curves(self):
        history = self.history
        acc = history.history['categorical_accuracy']
        val_acc = history.history['val_categorical_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs_range = range(1, len(acc) + 1)

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        axs[0].plot(epochs_range, loss, label='Train Loss', marker='o')
        axs[0].plot(epochs_range, val_loss, label='Val Loss', marker='x')
        axs[0].set_title('Loss vs Epochs')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()

        axs[1].plot(epochs_range, acc, label='Train Accuracy', marker='o')
        axs[1].plot(epochs_range, val_acc, label='Val Accuracy', marker='x')
        axs[1].set_title('Accuracy vs Epochs')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend()

        plt.tight_layout()
        plt.show()


# Dictionary of backbones
backbones = {
    "VGG16": VGG16,
    "InceptionV3": InceptionV3,
    "Xception": Xception,
    "DenseNet201": DenseNet201,
    "NASNetLarge": NASNetLarge,
    "ConvNeXtXLarge": ConvNeXtXLarge,
}

