import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
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

# Paths and labels
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

labels = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}
target_shape = (64, 64, 64)
import numpy as np
# Load and preprocess data for all regions (sum all regions)

X_all, y_all = [], []
X_region_1, y_region_1 = load_dataset(regions_paths[0], labels, target_shape) 
X_all = np.zeros_like(X_region_1)  # Same shape as X_region_1
y_all = np.zeros_like(y_region_1)
for region_path in regions_paths:
    X_region, y_region = load_dataset(region_path, labels, target_shape)
    X_all += X_region
    y_all=y_region
import os
import nibabel as nib
import numpy as np

# Create a directory to save the summed NIfTI files
output_dir = "Summed_NIfTI_Files"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Save each summed subject as a separate NIfTI file
for i in range(X_all.shape[0]):  # Iterate over subjects
    summed_mri = X_all[i]  # Extract single subject summed MRI
    nifti_img = nib.Nifti1Image(summed_mri, affine=np.eye(4))  # Convert to NIfTI format
    file_path = os.path.join(output_dir, f"subject_{i+1:03d}_summed.nii.gz")  # Naming format
    nib.save(nifti_img, file_path)  # Save the file
    print(f"Saved: {file_path}")  # Print confirmation

print("  X_all--      |||||||",X_all,"y_all--",y_all)
# Convert lists to numpy arrays and sum the regions for each sample
X = X_all[..., np.newaxis]  # Add a channel dimension
y = to_categorical(y_all, num_classes=len(labels))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1))
# Define CNN model
from tensorflow.keras.layers import Add, LeakyReLU
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Add
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv3D(32, kernel_size=(3, 3, 3), kernel_regularizer=l2(0.001), input_shape=input_shape),
        LeakyReLU(alpha=0.1),
        MaxPooling3D(pool_size=(2, 2, 2)),

        Conv3D(64, kernel_size=(3, 3, 3), kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.1),
        MaxPooling3D(pool_size=(2, 2, 2)),

        Conv3D(128, kernel_size=(3, 3, 3), kernel_regularizer=l2(0.001)),
        LeakyReLU(alpha=0.1),
        MaxPooling3D(pool_size=(2, 2, 2)),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),  # Reduced to prevent overfitting

        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
input_shape = X_train.shape[1:]
num_classes = len(labels)
model = create_cnn_model(input_shape, num_classes)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)

history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[early_stopping, reduce_lr])



# Create and train the model

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predictions to class indices
y_test_classes = np.argmax(y_test, axis=1)  # Convert true labels to class indices

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=labels.keys()))

# Confusion matrix (optional)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_classes, y_pred_classes))
# Save the model
model.save("mri_cnn_model_AugmentedData-summation_reg_mask_6_march_7_7_58pm.h5")
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
