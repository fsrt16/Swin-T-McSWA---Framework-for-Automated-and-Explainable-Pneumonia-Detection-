import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def preprocess_images(
    folder_path,
    img_size=(224, 224),
    grayscale=False,
    apply_clahe=False,
    verbose=True
):
    """
    Preprocess images from folder:
    - Load images from class-wise folders
    - Resize to target shape
    - Normalize to [0, 1]
    - Apply optional CLAHE
    - Encode labels to one-hot format

    Parameters:
    ----------
    folder_path : str
        Path to the dataset directory where subfolders are class names.
    img_size : tuple
        Target image size (height, width).
    grayscale : bool
        If True, load images in grayscale.
    apply_clahe : bool
        If True, apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    verbose : bool
        If True, print logs during preprocessing.

    Returns:
    -------
    images : np.ndarray
        Preprocessed image array.
    labels : np.ndarray
        One-hot encoded labels.
    label_encoder : LabelEncoder
        LabelEncoder object (can be used to decode class names).
    """
    images = []
    labels = []

    class_dirs = [
        d for d in os.listdir(folder_path)
        if os.path.isdir(os.path.join(folder_path, d))
    ]

    if verbose:
        print(f"[INFO] Found {len(class_dirs)} classes: {class_dirs}")

    for label in class_dirs:
        label_path = os.path.join(folder_path, label)
        image_files = os.listdir(label_path)

        for image_file in image_files:
            try:
                image_path = os.path.join(label_path, image_file)
                if grayscale:
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                else:
                    image = cv2.imread(image_path)

                if image is None:
                    if verbose:
                        print(f"[WARNING] Unable to read image: {image_path}")
                    continue

                image = cv2.resize(image, img_size)

                if grayscale:
                    if apply_clahe:
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        image = clahe.apply(image)
                    image = image[..., np.newaxis]  # Add channel dimension

                else:
                    if apply_clahe:
                        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        l_clahe = clahe.apply(l)
                        lab_clahe = cv2.merge((l_clahe, a, b))
                        image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

                image = image.astype('float32') / 255.0  # Normalize to [0, 1]
                images.append(image)
                labels.append(label)

            except Exception as e:
                if verbose:
                    print(f"[ERROR] Failed processing {image_file}: {e}")

    images = np.array(images)
    if verbose:
        print(f"[INFO] Processed {len(images)} images")

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    one_hot_labels = to_categorical(encoded_labels)

    if verbose:
        print(f"[INFO] Classes encoded as: {list(label_encoder.classes_)}")

    return images, one_hot_labels, label_encoder
