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
from monai.transforms import (
    EnsureChannelFirst,
    ScaleIntensity,
    Rand2DElastic,
    RandFlip,
    RandRotate,
    RandZoom,
    RandGaussianNoise,
    RandBiasField,
    RandAdjustContrast
)
from monai.transforms import Compose
C0 = []
C1 = []
C2 = []
C3 = []
c0 = 0
c1 = 0
c2 = 0
c3 = 0

def list_files_recursive(path='.'):
    global c0, c1, c2, c3  # Declare all counters as global
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            list_files_recursive(full_path)
        else:
            if "/C0/" in full_path.replace("\\", "/") and "lhippo_reg.nii.gz" in full_path:
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

# Specify the directory path you want to start from
directory_path = r'C:\Users\A.C\Documents\Final year project (FYP)\Subjects'

# Call the recursive function
list_files_recursive(directory_path)

# Print the results
print("The number of lhippo_reg.nii in c0 is:", c0)
print("The number of lthal_reg_mask.nii in c1 is:", c1)
print("The number of lthal_reg_mask.nii in c2 is:", c2)
print("The number of lthal_reg_mask.nii in c3 is:", c3)
transform = Compose([
RandomAffine(scales=(0, 100), degrees=10, translation=(1.28, 1.28,1.28), p=0.3),
RandomFlip(axes=(0, 1, 2), p=0.5),
RandomNoise(mean=0.0, std=0.1742, p=0.3),
RandomGamma(log_gamma=(-0.1, 0.1), p=0.3) # Random gamma correction
])


# Define target count

import os
import torchio as tio
import nibabel as nib

# Define the target number of images for each list
TARGET_COUNT = 50

# Directories for augmented data
output_dir = r'C:\Users\A.C\Documents\Final year project (FYP)\AugmentedData-lhippo_reg'
os.makedirs(output_dir, exist_ok=True)
def normalize_image(image):
    """Normalize the image intensity to the range [0, 1]."""
    data = image.data.numpy()
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Clip values to remove extreme outliers
    data = np.clip(data, min_val, max_val)
    
    # Normalize to [0, 1]
    normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
    print(f"Normalization Applied: Min={normalized_data.min()}, Max={normalized_data.max()}")
    normalized_tensor = torch.from_numpy(normalized_data)  # Convert NumPy to Tensor
    return image.__class__(normalized_tensor, affine=image.affine)
def save_augmented_image(augmented, output_path,original_):
    """Save the augmented image after normalization."""
    # Convert MetaTensor to a NIfTI image
    img_data = augmented.data.numpy()  # Convert tensor to NumPy array
    affine = original_.affine  # Get the affine (spatial information)

    # Create a NIfTI image
    nii_img = nib.Nifti1Image(img_data, affine)

    # Save the image to the specified output path
    nib.save(nii_img, output_path)
    print(f"Saved augmented image to {output_path}")
def augment_images(image_paths, target_count, folder_name):
    folder_path = os.path.join(output_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Calculate the number of augmentations needed
    current_count = len(image_paths)
    if current_count >= target_count:
        print(f"{folder_name}: Already has {current_count} images, skipping augmentation.")
        for i, image_path in enumerate(image_paths[:target_count]):  # Limit to first 50 images
            base_name = os.path.basename(image_path).replace('.nii.gz', '')
            full_directory_path = os.path.dirname(image_path)
            relative_directory = os.path.relpath(full_directory_path, start=r'C:\Users\A.C\Documents\Final year project (FYP)\Subjects')
            directory_name = relative_directory.replace(os.sep, '_')

             # Save the original (base) image in the output directory
            base_output_path = os.path.join(folder_path, f"{directory_name}_{base_name}.nii.gz")
            if not os.path.exists(base_output_path):  # Avoid duplicate saves
                shutil.copy(image_path, base_output_path)
                print(f"Copied base image: {base_output_path}")
    elif current_count==0:
        print("No imagees found for augmentation")
        return

    needed_count = target_count - current_count
    augment_per_image = needed_count // current_count
    extra_augments = needed_count % current_count

    # Perform augmentations
    augmented_count = 0
    for i, image_path in enumerate(image_paths):
        #mri_data = tio.ScalarImage(image_path)
        #mri = mri_data.data
        nifti_image = nib.load(image_path)
        image_data = nifti_image.get_fdata()
        affine = nifti_image.affine
        binary_image_data = (image_data > 0).astype(np.uint8)
        base_name = os.path.basename(image_path).replace('.nii.gz', '')
    
    # Extract the full directory path (this will be used in the augmented filename)
    
    # Convert directory path to a format that can be used in a filename 
    
    # Extract the full directory path
        full_directory_path = os.path.dirname(image_path)
    
    # Get the directory structure relative to the base directory (i.e., 'Subjects')
        relative_directory = os.path.relpath(full_directory_path, start=r'C:\Users\A.C\Documents\Final year project (FYP)\Subjects')
        directory_name = relative_directory.replace(os.sep, '_')
        base_output_path = os.path.join(folder_path, f"{directory_name}_{base_name}_augmented_{i}_.nii.gz")
        if not os.path.exists(base_output_path):  # Avoid duplicate saves
            shutil.copy(image_path, base_output_path) 
        scaling_factor = 2
        rotation_angle = 45
        print(image_path,"base  name :",base_name)
        for j in range(augment_per_image + (1 if i < extra_augments else 0)):
            #augmented = transform(mri)
        #    augmented = normalize_image(augmented)
         #   augmented= np.clip(augmented, 0, 1)
            output_path = os.path.join(folder_path, f"{directory_name}_{base_name}_augmented_{i}_{j}.nii.gz")
            if j % 2 == 0:  # Perform scaling on even iterations
                scaling_factor += 0.01
                scaled_image_data = rescale(binary_image_data, scaling_factor, order=0, mode='symmetric', cval=0, preserve_range=True)
        
        # Create a NIfTI image for scaled data
                scaled_nifti = nib.Nifti1Image(scaled_image_data, nifti_image.affine)
        
        # Save the scaled image
                nib.save(scaled_nifti, output_path)

            else:  # Perform rotation on odd iterations
                rotation_angle += 1
                rotated_image_data = rotate(binary_image_data, rotation_angle, axes=(0, 1), reshape=True, mode='nearest', cval=0)
        
        # Create a NIfTI image for rotated data
                rotated_nifti = nib.Nifti1Image(rotated_image_data, nifti_image.affine)
        
        # Save the rotated image
                nib.save(rotated_nifti, output_path)
        
            #save_augmented_image(augmented, output_path, mri_data)
            augmented_count += 1

    print(f"{folder_name}: Created {augmented_count} augmented images. Total: {target_count}")

#Augment images for each list
augment_images(C0, TARGET_COUNT, "C0")
augment_images(C1, TARGET_COUNT, "C1")
augment_images(C2, TARGET_COUNT, "C2")
augment_images(C3, TARGET_COUNT, "C3")

