from flask import Flask, render_template, request, redirect, url_for
import os
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
from flask import Flask, request, render_template
import os
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import numpy as np
from flask import Flask, request, render_template
import os
import numpy as np
import scipy.io
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from sklearn.preprocessing import StandardScaler
sys.stdout.reconfigure(encoding='utf-8')
app = Flask(__name__)
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
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
PROCESSED_FOLDER = "static/images"
STATIC_FOLDER = "static"
MODEL_PATH = r"C:\Users\A.C\Documents\Final year project (FYP)\EEG_ADHD_Model.h5"  # Update with the actual path to your trained model
# Load the trained model once when the server starts
import os
if os.path.exists(MODEL_PATH):
    model_ = load_model(MODEL_PATH)
else:
    print(f"Model file not found at {MODEL_PATH}")
model_ = load_model(MODEL_PATH)
# Ensure necessary directories exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def login():
    return render_template('login.html')

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash



# This will simulate a simple database for storing user credentials
# In a real-world scenario, replace this with a proper database (SQLite, MySQL, etc.)
users_db = {}

# Registration Route (GET and POST)
@app.route('/register', methods=['GET', 'POST'])
def register_user():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if the username already exists
        if username in users_db:
            return "Username already exists. Please choose another.", 400
        
        # Hash the password before storing it
        hashed_password = generate_password_hash(password)
        
        # Store the user data in the simulated "database"
        users_db[username] = {
            'email': email,
            'password': hashed_password
        }
        
        # Redirect to login page after successful registration
        return redirect(url_for('login'))
    
    return render_template('register.html')  # Render registration form on GET request

# Login Route (GET and POST)
@app.route('/login', methods=['GET', 'POST'])
def login_user():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if the username exists in the database
        if username in users_db:
            # Retrieve the stored hashed password
            stored_hashed_password = users_db[username]['password']
            
            # Check if the password matches the stored hashed password
            if check_password_hash(stored_hashed_password, password):
                return render_template('dashboard.html')  # Render dashboard on successful login
            else:
                return 'INCORRECT PASSWORD ', 403  # Incorrect password
        else:
            return 'USERNAME NOT FOUND ', 403  # Username not found
    
    return render_template('login.html')  # Render login form on GET request


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/EEG-classification')
def eeg_classification():
    return render_template('EEG-classification.html')
@app.route('/create-account')
def create_account():
    return render_template('create_account.html')

#@app.route('/register', methods=['POST'])
#def register_user():
 #   username = request.form['username']
  #  email = request.form['email']
   # password = request.form['password']
    ## Add registration logic (e.g., save to database)
    #return redirect(url_for('login'))
def load_eeg_mat(file_path, data_key):
    mat_data = scipy.io.loadmat(file_path)
    eeg_data = mat_data[data_key]  # Replace with the correct variable name in .mat file
    return pd.DataFrame(eeg_data)
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)
def get_data_key(file_path):
    """ Extract the key name dynamically from the .mat file """
    mat_data = scipy.io.loadmat(file_path)
    valid_keys = [key for key in mat_data.keys() if not key.startswith("__")]
    
    if len(valid_keys) == 1:
        return valid_keys[0]  # If only one valid key exists, return it
    elif len(valid_keys) > 1:
        print(f"Multiple keys found in {file_path}: {valid_keys}. Using the first one.")
        return valid_keys[0]  # Adjust logic if needed
    else:
        raise ValueError(f"No valid EEG data key found in {file_path}")
import joblib
file_path__=r'C:\Users\A.C\Documents\Final year project (FYP)\scaler.pkl'
# Load the pre-fitted scaler (this assumes you've saved the scaler as 'scaler.pkl')
scaler = joblib.load(file_path__)   
print(scaler)
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
    predictions = model_.predict(reshaped_new_eeg)
    predicted_classes = (predictions > 0.5).astype('int32')[0]

    # Convert predicted classes to human-readable format
    results = ["ADHD" if pred == 1 else "Healthy" for pred in predicted_classes]
    return results
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
def plot_eeg(df, title, file_name):
    """Generate and save EEG plots."""
    fig, axs = plt.subplots(df.shape[1], 1, figsize=(10, 5), sharex=True)
    for i, ax in enumerate(axs):
        ax.plot(df.iloc[:, i], color="black")
        ax.set_ylabel(f'Ch {i+1}', rotation=0)
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticks([])
        ax.spines[["top", "bottom", "left", "right"]].set_visible(False)
    
    fig.suptitle(title, fontsize=16)
    save_path = os.path.join(PROCESSED_FOLDER, file_name)
    plt.savefig(save_path)
    plt.close()
    
    return save_path
@app.route("/", methods=["GET", "POST"])
def index():
    raw_image = None
    processed_image = None
    prediction = None

    if request.method == "POST":
        uploaded_file = request.files["file"]
        
        if uploaded_file:
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(file_path)
            # Process EEG file
            data_key = get_data_key(file_path)  # Update based on your .mat file structure
            raw_eeg = load_eeg_mat(file_path, data_key)
            raw_image = plot_eeg(raw_eeg, "Raw EEG Data", "raw_eeg.png")
            print("file apth and datakey :",file_path,data_key)
            denoised_eeg = denoise_eeg(raw_eeg)
            processed_image = plot_eeg(denoised_eeg, "Denoised EEG Data", "processed_eeg.png")

            # Predict ADHD classification
            prediction = predict_new_data(file_path, data_key)
            print("prediction of the disease is ",prediction)
    return render_template("EEG-classification.html", raw_image=raw_image, processed_image=processed_image, prediction=prediction)
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        print("filepath :",filepath)
        # --- Process the file with the ML Model ---
          # Path to the saved model
        mri_scan_path =filepath  # Replace with the path to your MRI scan
        if "lamyg_reg_mask" in filepath:
            model_path = "mri_cnn_model_AugmentedData-lamyg_reg_mask.h5"
        elif "lcaud_reg_mask"in filepath:
            model_path="mri_cnn_model_AugmentedData-lcaud_reg_mask.h5"
        elif "lhippo_reg"in filepath:
            model_path="mri_cnn_model_AugmentedData-lhippo_reg.h5"
        elif "lhippo_reg_mask"in filepath:
            model_path="mri_cnn_model_AugmentedData-lhippo_region_mask.h5"
        elif "lputa_reg_mask"in filepath:
            model_path="mri_cnn_model_AugmentedData-lputa_reg_mask.h5"
        elif "ramyg_reg_mask"in filepath:
            model_path="mri_cnn_model_AugmentedData-ramyg_reg_mask.h5"
        elif "rcaud_reg_mask"in filepath:
            model_path="mri_cnn_model_AugmentedData-rcaud_reg_mask.h5"
        elif "rhippo_reg_mask"in filepath:
            model_path="mri_cnn_model_AugmentedData-rhippo-region-mask.h5"
        elif "rputa_reg_mask"in filepath:
            model_path="mri_cnn_model_AugmentedData-rputa_reg_mask.h5"

        labels = {"C0": 0, "C1": 1, "C2": 2, "C3": 3}
# Open the file and write data

    # Write initial text
    # Predict the class for C0
    
        predicted_class = predict_mri_class(model_path, mri_scan_path, labels)  # Replace with your actual function
        
        # --- Set the solution dynamically ---
        solution = f"Diagnosis complete. The results are verified as {predicted_class}."
        print(f"model path : 0\n",model_path,"predicted class",predicted_class,"file_path",filepath)
        return render_template('dashboard.html', solution=solution)
if __name__ == '__main__':
    app.run(debug=True)
