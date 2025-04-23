import os
import numpy as np
import nibabel as nib
from mayavi import mlab

# Function to load and visualize MRI scan in 3D
def visualize_mri_3d(nifti_file):
    # Load the NIfTI image
    img = nib.load(nifti_file)
    data = img.get_fdata()

    # Normalize data for better visualization
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # Create a 3D figure using Mayavi
    mlab.figure(size=(800, 600), bgcolor=(0, 0, 0))  # Black background

    # Render the MRI volume
    vol = mlab.contour3d(
        data,
        contours=6,  # Number of contour levels
        opacity=0.4,  # Transparency
    )

    # Add interactive controls
    mlab.colorbar(vol, title="", orientation="vertical")
    mlab.title("")
    mlab.view(azimuth=120, elevation=80)  # Set initial view

    # Show the figure
    mlab.show()

# Example usage
nifti_file = r"C:\Users\A.C\Documents\sample_10_sets_dataset_for prediction\Summed_Image_C1_1-.nii.gz" 
if os.path.exists(nifti_file):
    visualize_mri_3d(nifti_file)
else:
    print(f"File not found: {nifti_file}")
nifti_file = r"C:\Users\A.C\Documents\sample_10_sets_dataset_for prediction\Summed_Image_C2_2-.nii.gz"
if os.path.exists(nifti_file):
    visualize_mri_3d(nifti_file)
else:
    print(f"File not found: {nifti_file}")
nifti_file = r"C:\Users\A.C\Documents\sample_10_sets_dataset_for prediction\Summed_Image_C3_3-.nii.gz" # Replace with your NIfTI file path
if os.path.exists(nifti_file):
    visualize_mri_3d(nifti_file)
else:
    print(f"File not found: {nifti_file}")
import numpy as np
import nibabel as nib
from vedo import Volume, show

def display_mri_3d(file_path):
    # Load NIfTI image
    img = nib.load(file_path)
    data = img.get_fdata()

    # Normalize the image to [0,255] for better visualization
    data = np.nan_to_num(data)  # Replace NaN values with 0
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    data = data.astype(np.uint8)

    # Create a 3D volume
    vol = Volume(data)
    
    # Apply colormap and opacity settings to enhance visualization
    vol.cmap("gray")  # Use a grayscale colormap
    vol.alpha([0.0, 0.1, 0.5, 0.8, 1])  # Adjust transparency levels
    vol.add_scalarbar("Intensity")  # Add color intensity scale

    # Display the 3D MRI scan interactively
    show(vol, axes=1, title="3D MRI Visualization with vedo")

# Example usage



# Example usage
file_path = r"C:\Users\A.C\Documents\Final year project (FYP)\HTML\Summed_NIfTI_Files\subject_001_summed.nii.gz"  # Change this to your file path
display_mri_3d(file_path)
