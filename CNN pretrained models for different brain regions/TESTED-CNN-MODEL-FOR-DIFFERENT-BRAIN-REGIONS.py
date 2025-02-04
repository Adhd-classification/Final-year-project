import os
import numpy as np
import nibabel as nib
from tensorflow.keras.models import load_model
import torchio as tio
# Let's load some other packages we need
import os
import numpy as np
import matplotlib.pyplot as plt
import torchio as tio
import shutil
import torch
import nibabel as nib
import numpy as np
from scipy.ndimage import rotate
from skimage.transform import rescale
import matplotlib.pyplot as plt
#matplotlib inline
import nibabel as nib # common way of importing nibabel
from torchio.transforms import Compose, RandomAffine, RandomFlip, RandomNoise, RandomGamma
# Preprocessing function for a single MRI scan
from skimage.transform import resize
import nibabel as nib
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')
def preprocess_mri_scan(file_path, target_shape=(64, 64, 64)):
    """
    Preprocess a single MRI scan for prediction.
    Args:
        file_path (str): Path to the NIfTI file.
        target_shape (tuple): Target shape for resizing (depth, height, width).
    Returns:
        np.ndarray: Preprocessed MRI scan ready for prediction.
    """
    img = nib.load(file_path).get_fdata()
    img_resized = resize(img, target_shape, mode='constant', anti_aliasing=True) 
    return img_resized[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

# Prediction function
def predict_mri_class(model_path, file_path, labels, target_shape=(64, 64, 64)):
    """
    Predict the class of a given MRI scan.
    Args:
        model_path (str): Path to the saved model.
        file_path (str): Path to the MRI scan file.
        labels (dict): Dictionary mapping class indices to labels.
        target_shape (tuple): Target shape for resizing.
    Returns:
        str: Predicted class label.
    """
    # Load the trained model
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Preprocess the MRI scan
    preprocessed_scan = preprocess_mri_scan(file_path, target_shape)
    
    # Predict the class
    predictions = model.predict(preprocessed_scan)
    
    predicted_class_index = np.argmax(predictions)
    
    # Get the corresponding label
    label_map = {v: k for k, v in labels.items()}
    predicted_label = label_map[predicted_class_index]
    
    return predicted_label
C0 = []
C1 = []
C2 = []
C3 = []
c0 = 0
c1 = 0
c2 = 0
c3 = 0
# Example usage
def list_files_recursive(path='.'):
    global c0, c1, c2, c3  # Declare all counters as global
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            list_files_recursive(full_path)
        else:
            if "C0" in full_path and "lhippo_reg.nii.gz" in full_path:
                C0.append(full_path)
                print(full_path)
                c0 += 1  # Correctly increment the counter
            elif "C1" in full_path and "lhippo_reg.nii.gz" in full_path:
                C1.append(full_path)
                print(full_path)
                c1 += 1
            elif "C2" in full_path and "lhippo_reg.nii.gz" in full_path:
                C2.append(full_path)
                print(full_path)
                c2 += 1
            elif "C3" in full_path and "lhippo_reg.nii.gz" in full_path:
                C3.append(full_path)
                print(full_path)
                c3 += 1
if __name__ == "__main__":
# Specify the directory path you want to start from
    directory_path = r'C:\Users\A.C\Documents\Final year project (FYP)\Subjects'

# Call the recursive function
    list_files_recursive(directory_path)
    model_path = "mri_cnn_model.h5"  # Path to the saved model
    mri_scan_path = "path_to_sample_mri.nii.gz"  # Replace with the path to your MRI scan
    labels = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}

# Open the file and write data

    # Write initial text
    print(f"The predicted class for the input MRI scan is: 0\n")

    # Predict the class for C0
    
    for path in C0[:10]:
        predicted_class = predict_mri_class(model_path, path, labels)
        print(f"The predicted class for the input MRI scan is_: {predicted_class}\n")
    print("-------------------------------------------------------------\n")

    # Predict the class for C1
    for path in C1[:10]:
        predicted_class = predict_mri_class(model_path, path, labels)
        print(f"The predicted class for the input MRI scan is_: {predicted_class}\n")
    print("-------------------------------------------------------------\n")

    # Predict the class for C2
    for path in C2[:10]:
        predicted_class = predict_mri_class(model_path, path, labels)
        print(f"The predicted class for the input MRI scan is_: {predicted_class}\n")
    print("-------------------------------------------------------------\n")

    # Predict the class for C3
    for path in C3[:10]:
        predicted_class = predict_mri_class(model_path, path, labels)
        print(f"The predicted class for the input MRI scan is_: {predicted_class}\n")
    print ("-------------------------------------------------------------\n")