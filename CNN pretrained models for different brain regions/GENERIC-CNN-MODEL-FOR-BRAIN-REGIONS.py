import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout 
from tensorflow.keras.utils import to_categorical
from skimage.transform import resize
import sys
from sklearn.metrics import classification_report, confusion_matrix
sys.stdout.reconfigure(encoding='utf-8')
import matplotlib.pyplot as plt 
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
base_dir = r"C:\Users\A.C\Documents\Final year project (FYP)\AugmentedData-lthal_reg_mask"  # Replace with the correct path
labels = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}
target_shape = (64, 64, 64)

# Load and split data
X, y = load_dataset(base_dir, labels, target_shape)
X = X[..., np.newaxis]  # Add a channel dimension
y = to_categorical(y, num_classes=len(labels))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1))

# Define CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=input_shape),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
        MaxPooling3D(pool_size=(2, 2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
input_shape = X_train.shape[1:]
num_classes = len(labels)
model = create_cnn_model(input_shape, num_classes)

history =model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2)

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
model.save("mri_cnn_model_AugmentedData-lthal_reg_mask.h5")
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
