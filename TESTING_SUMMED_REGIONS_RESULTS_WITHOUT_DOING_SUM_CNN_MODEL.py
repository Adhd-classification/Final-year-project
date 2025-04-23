
import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from skimage.transform import resize
import zipfile
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
# Define the directory path
directory_path = r'C:\Users\A.C\Documents\sample_10_sets_dataset_for prediction'

# Initialize an empty list to store the file paths
regions_path = []

# Iterate through all files in the directory
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    
    # Check if it's a file (not a directory)
    if os.path.isfile(file_path):
        # Append the file path to regions_path
        regions_path.append(file_path)

# Print the list of file paths
# Load and preprocess images (without resizing or normalizing)
X_test = []

for file_path in regions_path:
    # Load the NIfTI file using nibabel
    img = nib.load(file_path)
    
    # Get the image data as a numpy array
    img_data = img.get_fdata()
    
    # Optionally, you can add a check to make sure the shape is correct for your model's input
    # For example, ensure that the images are 3D and have a specific shape
    # Check if the images need to be reshaped to fit the model's input requirements
    # img_data = np.expand_dims(img_data, axis=-1)  # Add channel dimension if necessary
    X_test = []
    X_test.append(img_data)
    # Append the image data to the X_test list
    break

# Convert X_test to a numpy array
    X_test = []
    X_test.append(img_data)
X_test = np.array(X_test)

# Load trained model
model_path = r"C:\Users\A.C\Documents\Final year project (FYP)\HTML\mri_cnn_model_AugmentedData-rthal_reg_mask.h5"
model = tf.keras.models.load_model(model_path)

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
print("Type of y_pred_classes:", type(y_pred_classes))
# Print predictions
print("Predicted Classes:", y_pred_classes)
batch_size = 50
for i in range(0, len(y_pred_classes), batch_size):
    print(y_pred_classes[i:i + batch_size])