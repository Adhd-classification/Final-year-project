import scipy.io as sio
import numpy as np
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
import pywt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Function to load EEG data from .mat files
def load_eeg_mat(file_path, data_key):
    mat_data = scipy.io.loadmat(file_path)
    eeg_data = mat_data[data_key]  # Replace with the correct variable name in .mat file
    return pd.DataFrame(eeg_data)

# Function to plot EEG data
def plot_eeg(df, title, file_name):
    fig, axs = plt.subplots(df.shape[1], 1, figsize=(30, 15), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(df.iloc[:, i], color="black")
        ax.set_ylabel(f'Ch {i+1}', rotation=0)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines[["top", "bottom", "left", "right"]].set_visible(False)
    fig.suptitle(title, fontsize=16)
    plt.savefig(file_name)
#   plt.show()

# Function for MAD-based threshold estimation
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

# Function to denoise EEG data using wavelet transformation
def denoise_eeg(df, wavelet='db8', level=1, threshold_factor=0.5):
    ret = pd.DataFrame(index=df.index, columns=df.columns)
    
    for col in df.columns:
        # Perform Wavelet decomposition
        coeff = pywt.wavedec(df[col], wavelet, mode="per")
        
        # Estimate noise level and calculate threshold
        sigma = (1 / 0.6745) * maddest(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(df))) * threshold_factor
        
        # Apply thresholding to detail coefficients
        coeff[1:] = [pywt.threshold(c, value=uthresh, mode='hard') for c in coeff[1:]]
        
        # Reconstruct the signal
        reconstructed_signal = pywt.waverec(coeff, wavelet, mode='per')
        
        # Ensure that the reconstructed signal has the same length as the original
        if len(reconstructed_signal) > len(df):
            reconstructed_signal = reconstructed_signal[:len(df)]
        elif len(reconstructed_signal) < len(df):
            reconstructed_signal = np.pad(reconstructed_signal, (0, len(df) - len(reconstructed_signal)), 'constant')
        
        # Store the denoised signal in the DataFrame
        ret[col] = reconstructed_signal
    
    return ret

# Function to process EEG data
def process_eeg(file_path, data_key):
    # Load raw EEG data from .mat file
    raw_eeg = load_eeg_mat(file_path, data_key)

    # Perform wavelet denoising
    denoised_eeg = denoise_eeg(raw_eeg, wavelet='db8', level=1, threshold_factor=0.5)
    

    return denoised_eeg
def process_eeg_(file_path, data_key):
    # Load raw EEG data from .mat file
    raw_eeg = load_eeg_mat(file_path, data_key)
    file=data_key
    file_name_=file+"-RAW"+".png"
    # Perform wavelet denoising
    plot_eeg(raw_eeg, title="Raw EEG Data", file_name=file_name_)

    # Perform wavelet denoising
    denoised_eeg = denoise_eeg(raw_eeg, wavelet='db8', level=1, threshold_factor=0.5)
    file_name_=file+"-PREPROCESSED"+".png"
    plot_eeg(denoised_eeg, title="Denoised EEG Data", file_name=file_name_)
    

    return denoised_eeg

# Load data from .mat files for training
def load_data():
    nifti_files = []
    X = []
    y = []
    TOTAL_SIZE = 0
    for root, dirs, files in os.walk(r".\EEG-Dataset"):
        for file in files:
            if file.endswith(".mat"):
                nifti_files.append(os.path.join(root, file))
                file_path = os.path.join(root, file)
                file_, file_extension = os.path.splitext(file)
                data_key = file_  # Replace with the correct key in your .mat file containing EEG data
                denoised_eeg_ = process_eeg(file_path, data_key)
                data = denoised_eeg_
                if "ADHD_part1" in file_path or "ADHD_part2" in file_path:
                    X.append(data)
                    y.extend([1] * data.shape[0])  # 1 for ADHD
                elif "Control_part1" in file_path or "Control_part2" in file_path:
                    X.append(data)
                    y.extend([0] * data.shape[0])  # 0 for Control
    return np.concatenate(X), np.array(y)

# Load training data
X, y = load_data()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=32))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001)

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                    validation_split=0.2, callbacks=[early_stopping])

# Evaluate model
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype('int32')
print("Accuracy:", accuracy_score(y_test, y_pred_class))
print("Classification Report:")
print(classification_report(y_test, y_pred_class))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_class))

# ----- New Data Prediction Section -----
def predict_new_data(new_file_path, data_key):
    # Load the new EEG data
    new_eeg_data = load_eeg_mat(new_file_path, data_key)

    # Denoise the new EEG data
    denoised_new_eeg = denoise_eeg(new_eeg_data)

    # Scale the data using the already fitted scaler
    scaled_new_eeg = scaler.transform(denoised_new_eeg)

    # Reshape for LSTM input
    reshaped_new_eeg = scaled_new_eeg.reshape(scaled_new_eeg.shape[0], scaled_new_eeg.shape[1], 1)

    # Make predictions
    predictions = model.predict(reshaped_new_eeg)
    predicted_classes = (predictions > 0.5).astype('int32')

    # Convert predicted classes to human-readable format
    results = ["ADHD" if pred == 1 else "Healthy" for pred in predicted_classes]
    return results
mat_files=[]
with open("file_name.txt", mode="wt") as f:
    f.write("")
for root, dirs, files in os.walk(r".\EEG-Dataset"):
        for file in files:
            if file.endswith(".mat"):
                mat_files.append(os.path.join(root, file))
                file_path = os.path.join(root, file)
                file_, file_extension = os.path.splitext(file)
                process_eeg_(file_path, file_)
                predictions = predict_new_data(file_path,file_)
                with open("file_name.txt", "a") as file_real:
                # Write the new data to the file
                    for file__, prediction in zip([file_path], predictions):
                        file_real.write(f"File: {file__}, Prediction: {prediction}\n")
def plot_accuracy_curves(history):
    # Extract accuracy and loss values from the training history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    # Plot training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # Show the plots
    plt.tight_layout()
    plt.show()

# Plot the accuracy and loss curves
plot_accuracy_curves(history)
# Example usage for new data
  # Update with the actual key in the .mat file

# Display the predictions in the terminal