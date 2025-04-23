import os
import numpy as np
import nibabel as nib
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import load_model

import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from skimage.transform import resize
import tensorflow as tf
import sys
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Ensure UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

# Define a function to load and preprocess NIfTI files
def load_data_from_directory(directory_path, label, target_shape=(64, 64, 64)):
    X, y = [], []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".gz"):
            file_path = os.path.join(directory_path, file_name)
            img = nib.load(file_path).get_fdata()
            img_resized = resize(img, target_shape, mode='constant', anti_aliasing=True)  # Resize without normalizing
            X.append(img_resized)
            y.append(label)
    return X, y

# Load data
def load_dataset(base_dir, labels, target_shape):
    X, y = [], []
    for class_name, label in labels.items():
        class_path = os.path.join(base_dir, class_name)
        X_class, y_class = load_data_from_directory(class_path, label, target_shape)
        X.extend(X_class)
        y.extend(y_class)
    return np.array(X), np.array(y)

# Function to predict for disorder
def predict_for_disorder():
    """
    Function to select specific samples from different regions, combine them,
    and use a pre-trained model to make predictions.

    The function selects the first 10 elements, skips the next 40, and repeats
    this pattern across the dataset.
    """
    # Define paths to region directories and class labels
    regions_paths = [
        r"C:\Users\A.C\Documents\Final year project (FYP)\AugmentedData-lamyg_reg_mask",
        r"C:\Users\A.C\Documents\Final year project (FYP)\AugmentedData-lcaud_reg_mask",
        r"C:\Users\A.C\Documents\Final year project (FYP)\AugmentedData-lhippo-region-mask",
        r"C:\Users\A.C\Documents\Final year project (FYP)\AugmentedData-lputa_reg_mask",
        r"C:\Users\A.C\Documents\Final year project (FYP)\AugmentedData-ramyg_reg_mask",
        r"C:\Users\A.C\Documents\Final year project (FYP)\AugmentedData-rcaud_reg_mask",
        r"C:\Users\A.C\Documents\Final year project (FYP)\AugmentedData-rhippo-region-mask",
        r"C:\Users\A.C\Documents\Final year project (FYP)\AugmentedData-rputa_reg_mask"
    ]
    
    # Dictionary of class labels
    labels = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}
    target_shape = (64, 64, 64)  # Target shape for resizing images
    X_all, y_all = [], []
    X_region_1, y_region_1 = load_dataset(regions_paths[0], labels, target_shape) 
    X_all = np.zeros_like(X_region_1)  # Same shape as X_region_1
    y_all = np.zeros_like(y_region_1)
    # Initialize empty lists to store combined data

    # Loop through each region path and load the data from each region
    for region_path in regions_paths:
        X_region, y_region = load_dataset(region_path, labels, target_shape)  # Load data for this region
        X_all+=X_region  # Add images from this region to the list
        y_all=y_region # Add labels from this region to the list

    # Convert the lists to numpy arrays for processing
    X_all = np.array(X_all)  # Array of images
    y_all = np.array(y_all)  # Array of labels

    # Initialize lists to store selected samples
    X_selected, y_selected = [], []

    # Select the first 10 elements, skip the next 40, repeat this pattern
    for i in range(0, len(X_all), 50):  # The step size of 50 includes 10 selected samples + 40 skipped samples
        # Select the next 10 samples
        X_selected.extend(X_all[i:i+10])  # Add 10 selected samples
        y_selected.extend(y_all[i:i+10])  # Add the corresponding labels

    # Convert selected data to numpy arrays
    X_selected = np.array(X_selected)  # Array of selected images
    y_selected = np.array(y_selected)  # Array of selected labels

    # Add a channel dimension to match CNN input requirements
    X_selected = X_selected[..., np.newaxis]  # Add channel dimension to images

    # One-hot encode the labels for classification
    y_selected = to_categorical(y_selected, num_classes=len(labels))  # Convert labels to one-hot encoding
    model_path=r"model-path"
    # Predict using the pre-trained model
    model = load_model(model_path)
    print("Model loaded successfully.")

    y_pred = model.predict(X_selected)  # Predict the classes of the selected data
    y_pred_classes = np.argmax(y_pred, axis=1)  # Get the predicted class labels

    # Print prediction results for each selected sample
    print("\nPrediction Results:")
    for idx, pred_class in enumerate(y_pred_classes):
        print(f"Sample {idx + 1}: Predicted Class = {list(labels.keys())[pred_class]}")  # Print class label

# Call the prediction function
predict_for_disorder()  # Run the prediction function

