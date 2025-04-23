import os
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from skimage.transform import resize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter
# ----------------------------- Load & Build Model -----------------------------
model_path = r"C:\Users\A.C\Documents\finalyearprojectgithub\CODE_WITH_CAM\rupta_reg_mask.h5"
model = load_model(model_path)

# Force model to build by specifying the input shape
model.build(input_shape=(None, 64, 64, 64, 1))  # Set the correct input shape here

# --------------------------- Preprocessing Function ---------------------------
def preprocess_mri(img_path, target_shape=(64, 64, 64)):
    img = nib.load(img_path).get_fdata()
    img = gaussian_filter(img, sigma=1.0)
    img_resized = resize(img, target_shape, mode='constant', anti_aliasing=True)
    img_resized = np.expand_dims(img_resized, axis=-1)  # (64,64,64,1)
    img_resized = np.expand_dims(img_resized, axis=0)   # (1,64,64,64,1)
    return img_resized

# ----------------------------- CAM Generation ---------------------------------
from tensorflow.keras.models import Model
from skimage.transform import resize
import numpy as np

def generate_cam(model, img_array, conv_layer_name, final_dense_name, target_shape=(64, 64, 64)):
    """
    Generate a 3D Class Activation Map (CAM) for the predicted class.

    Args:
        model: Trained Keras model
        img_array: Preprocessed input image of shape (1, D, H, W, 1)
        conv_layer_name: Name of the 3D convolutional layer to base the CAM on
        final_dense_name: Name of the final dense (classification) layer
        target_shape: Shape to resize the CAM to match input (default: 64x64x64)

    Returns:
        cam_resized: Resized CAM as a NumPy array
        predicted_class: Integer index of predicted class
    """

    conv_layer = model.get_layer(conv_layer_name)
    final_dense = model.get_layer(final_dense_name)

    # Create the CAM model
    cam_model = Model(inputs=model.inputs, outputs=[conv_layer.output, final_dense.output])
    conv_output, predictions = cam_model(img_array)
    conv_output = conv_output[0].numpy()  # shape: (D, H, W, C)
    predicted_class = np.argmax(predictions[0].numpy())

    # Get dense layer weights and class-specific weights
    weights = final_dense.get_weights()[0]  # shape: (num_features, num_classes)
    class_weights = weights[:, predicted_class]  # shape: (num_features,)

    # Match number of weights to channels (in case of mismatch)
    num_channels = conv_output.shape[-1]
    class_weights = class_weights[:num_channels]

    # Compute weighted sum of feature maps
    cam = np.zeros(conv_output.shape[:-1], dtype=np.float32)
    for i, w in enumerate(class_weights):
        cam += w * conv_output[..., i]

    # ReLU (keep positive only)
    cam = np.maximum(cam, 0)

    # Normalize
    cam_max = np.max(cam)
    if cam_max > 0:
        cam /= cam_max

    # Resize CAM to match input
    cam_resized = resize(cam, target_shape, mode='constant', anti_aliasing=True)

    # Diagnostics (optional)
    print(f" CAM generated: shape={cam_resized.shape}, max={np.max(cam_resized):.4f}, mean={np.mean(cam_resized):.4f}")

    return cam_resized, predicted_class

# ------------------------- Contributing Region Extraction ---------------------
def extract_contributing_regions(cam, threshold_percentile=95):
    """
    Returns a binary mask where CAM values are in the top percentile.
    """
    threshold_value = np.percentile(cam, threshold_percentile)
    contributing_mask = cam >= threshold_value
    return contributing_mask.astype(np.uint8)

# --------------------------- Overlay & Visualization --------------------------
def overlay_heatmap_on_mri_3d(mri_img, heatmap):
    mri_img = np.squeeze(mri_img, axis=(0, -1))  # (64,64,64)
    overlay = np.clip(mri_img + heatmap * 0.5, 0, 1)
    return overlay

def plot_3d_voxel(data, title="3D View"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    threshold = np.percentile(data, 95)
    x, y, z = np.where(data > threshold)
    ax.scatter(x, y, z, c=data[x, y, z], cmap='jet', alpha=0.6)
    plt.title(title)
    plt.show()

# ------------------------------ Main Pipeline ----------------------------------
mri_path = r"C:\Users\A.C\Documents\Final year project (FYP)\AugmentedData-rputa_reg_mask\C2\C2_NYU_0010005_1_rputa_reg_mask_augmented_0_.nii.gz"

# Use layer names from your model summary
conv_layer_name = "conv3d_2"
final_dense_name = "dense_1"

img_array = preprocess_mri(mri_path)

cam, predicted_class = generate_cam(model, img_array, conv_layer_name, final_dense_name)
print(f"Predicted class: {predicted_class}")

# Overlay for general visualization
overlay = overlay_heatmap_on_mri_3d(img_array, cam)

# ------------------------- Extract & Visualize Contributions -------------------
# 1. Get contributing binary mask
contributing_mask = extract_contributing_regions(cam, threshold_percentile=95)

# 2. Get only the contributing CAM voxels
contributing_voxels = cam * contributing_mask

# 3. Apply mask to MRI for anatomical localization
mri_only_contributing = np.squeeze(img_array, axis=(0, -1)) * contributing_mask

# ------------------------- Plot All Visualizations -----------------------------
import plotly.graph_objects as go

def plot_3d_mri_volume(volume_data, title="3D MRI Volume", threshold=0.001):
    """
    Visualize a 3D MRI scan using Plotly volumetric rendering.
    
    Parameters:
    - volume_data: 3D NumPy array (shape: [64, 64, 64])
    - threshold: minimum intensity to visualize
    """
    volume_data = np.squeeze(volume_data)  # Remove batch and channel dimensions if needed
    volume_data = (volume_data - np.min(volume_data)) / (np.max(volume_data) - np.min(volume_data))  # Normalize

    fig = go.Figure(data=go.Volume(
        x=np.linspace(0, volume_data.shape[0]-1, volume_data.shape[0]),
        y=np.linspace(0, volume_data.shape[1]-1, volume_data.shape[1]),
        z=np.linspace(0, volume_data.shape[2]-1, volume_data.shape[2]),
        value=volume_data.flatten(),
        isomin=threshold,
        isomax=volume_data.max(),
        opacity=0.1,  # adjust for better visibility
        surface_count=20,
        colorscale='Gray',
        caps=dict(x_show=False, y_show=False, z_show=False)
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=800,
        height=800
    )
    fig.show()
import os
import numpy as np
import nibabel as nib
from mayavi import mlab
def visualize_mri_3d(volume_data):
    """
    Visualize a 3D volume using Mayavi's contour3d.
    Accepts a 3D numpy array (e.g., [64,64,64]).
    """
    # Squeeze to remove batch and channel dimensions if present
    volume_data = np.squeeze(volume_data)  # from (1,64,64,64,1) → (64,64,64)

    if volume_data.ndim != 3:
        raise ValueError(f"Input volume must be 3D, but got shape {volume_data.shape}")

    # Create a 3D figure using Mayavi
    mlab.figure(size=(800, 600), bgcolor=(0, 0, 0))  # Black background

    # Render the MRI volume
    vol = mlab.contour3d(
        volume_data,
        contours=6,
        opacity=0.4,
    )

    mlab.colorbar(vol, title="", orientation="vertical")
    mlab.view(azimuth=120, elevation=80)
    mlab.title("3D MRI Visualization")
    mlab.show()

visualize_mri_3d(img_array)
visualize_mri_3d(cam)
visualize_mri_3d(overlay)
visualize_mri_3d(contributing_voxels)
visualize_mri_3d(mri_only_contributing)
plot_3d_mri_volume(img_array, title="3D MRI Scan")
plot_3d_voxel(cam, title="Full CAM Heatmap")
plot_3d_voxel(overlay, title="Overlay: MRI + CAM")
plot_3d_voxel(contributing_voxels, title="Top Contributing CAM Voxels")
plot_3d_voxel(mri_only_contributing, title="Anatomical Regions Contributing to Classification")