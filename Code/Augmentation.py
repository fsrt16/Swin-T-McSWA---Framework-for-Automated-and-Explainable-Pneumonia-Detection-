import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import plotly.express as px
import plotly.figure_factory as ff
from collections import defaultdict


def augment_data(X, Y, num_variations=7, class_names=None, visualize=False):
    """
    Augments input image dataset using random transformations.
    
    Parameters:
    -----------
    X : np.ndarray
        Array of images with shape (N, H, W, C)
    Y : np.ndarray
        Corresponding labels (one-hot or categorical).
    num_variations : int
        Number of new images to generate per original image.
    class_names : list or dict
        Optional mapping of class index to class name.
    visualize : bool
        If True, displays augmented image grid using Plotly.
    
    Returns:
    --------
    X_augmented : np.ndarray
        Augmented image dataset.
    Y_augmented : np.ndarray
        Corresponding labels.
    """
    print(f"[INFO] Starting augmentation with {num_variations} variations per image...")

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    X_augmented = []
    Y_augmented = []
    class_counts = defaultdict(int)

    for i in tqdm(range(len(X)), desc="Augmenting images"):
        x_sample = X[i].reshape((1,) + X[i].shape)
        y_sample = Y[i]

        label_idx = np.argmax(y_sample) if y_sample.ndim > 0 else y_sample
        class_label = class_names[label_idx] if class_names else str(label_idx)

        count = 0
        for batch in datagen.flow(x_sample, batch_size=1):
            X_augmented.append(batch[0])
            Y_augmented.append(y_sample)
            count += 1
            if count >= num_variations:
                break
        class_counts[class_label] += num_variations

    print("[INFO] Augmentation completed.")
    print("[INFO] Images added per class:")
    for cls, count in class_counts.items():
        print(f"  - {cls}: {count}")

    if visualize:
        _visualize_augmented_samples(np.array(X_augmented), Y_augmented, class_names)

    return np.array(X_augmented), np.array(Y_augmented)


def _visualize_augmented_samples(X_aug, Y_aug, class_names=None, n=25):
    """
    Helper function to visualize n augmented samples using Plotly.
    """
    print("[INFO] Visualizing sample augmentations...")
    import random
    import plotly.subplots as sp
    import plotly.graph_objects as go

    n = min(n, len(X_aug))
    indices = random.sample(range(len(X_aug)), n)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig = sp.make_subplots(rows=rows, cols=cols, subplot_titles=[
        class_names[np.argmax(Y_aug[i])] if class_names else str(np.argmax(Y_aug[i]))
        for i in indices
    ])

    for idx, i in enumerate(indices):
        row, col = divmod(idx, cols)
        fig.add_trace(
            go.Image(z=X_aug[i]),
            row=row + 1,
            col=col + 1
        )

    fig.update_layout(height=rows * 200, width=1000, title_text="Sample Augmented Images")
    fig.show()
