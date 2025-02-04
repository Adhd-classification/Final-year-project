import os
import shutil

def copy_contents(source_dir, dest_dir):
    """
    Copies the contents from source_dir to dest_dir.
    If the destination folder does not exist, it creates it.
    If files with the same name exist, they are overwritten.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print("warning")
    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        dest_item = os.path.join(dest_dir, item)
        if os.path.isdir(source_item):
            shutil.copytree(source_item, dest_item, dirs_exist_ok=True)
        else:
            shutil.copy2(source_item, dest_item)
          #  print("----------------source_item : ",source_item," -----------------dest_item  :", dest_item)
            print("oka")
def check_permissions(path):
    """
    Checks if the given path has read and write permissions.
    """
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return False

    read_permission = os.access(path, os.R_OK)  # Check read permission
    write_permission = os.access(path, os.W_OK)  # Check write permission

    print(f"Checking permissions for: {path}")
    print(f"Read permission: {'Yes' if read_permission else 'No'}")
    print(f"Write permission: {'Yes' if write_permission else 'No'}")

    return read_permission and write_permission
def pad_filenames(filenames):
    padded_filenames = set()
    for name in filenames:
        # Split into parts
        prefix, number, suffix = name.split('_')
        # Pad the number to 7 characters with leading zeros
        padded_number = number.zfill(7)
        # Reconstruct the filename
        padded_name = f"{prefix}_{padded_number}_{suffix}"
        padded_filenames.add(padded_name)
    return padded_filenames
def pad_filename(name):
    """
    Pads the number in the filename to 7 characters with leading zeros.
    
    Args:
        name (str): The filename to process (e.g., 'NYU_10062_1').
    
    Returns:
        str: The padded filename (e.g., 'NYU_0010062_1').
    """
    # Split into parts
    prefix, number, suffix = name.split('_')
    # Pad the number to 7 characters with leading zeros
    padded_number = number.zfill(7)
    # Reconstruct the filename
    padded_name = f"{prefix}_{padded_number}_{suffix}"
    return padded_name
def unpad_filename(name):
    """
    Removes leading zeros from the number in the filename.
    
    Args:
        name (str): The filename to process (e.g., 'NYU_0010062_1').
    
    Returns:
        str: The unpadded filename (e.g., 'NYU_10062_1').
    """
    # Split into parts
    prefix, number, suffix = name.split('_')
    # Remove leading zeros from the number
    unpadded_number = str(int(number))
    # Reconstruct the filename
    unpadded_name = f"{prefix}_{unpadded_number}_{suffix}"
    return unpadded_name
def copy_from_c01_to_c0_c1(subjects_dir):
    """    
    Searches for folders in C01 that match folders in C0 or C1,
    and copies the contents of C01's folders into those matching folders.
    """
    c01_path = os.path.join(subjects_dir, 'C03')
    print(c01_path)
    c0_path = os.path.join(subjects_dir, 'C0')
    print(c0_path)
    c1_path = os.path.join(subjects_dir, 'C3')
    print(c1_path)

    # Get folder names inside C01, C0, and C1
    c01_folders = set(os.listdir(c01_path))
    print(c01_folders)
    #c0_folders = set(os.listdir(c0_path))
    c1_folders = set(os.listdir(c1_path))
    print(c1_folders)
    c01__folders = pad_filenames(c01_folders)

# Print the result
    print(c01__folders)
    # Iterate through folders in C01
    for folder in c01__folders:
        name=unpad_filename(folder)
        c01_folder_path = os.path.join(c01_path, name)
    #    if folder in c0_folders:
     #       dest_path = os.path.join(c0_path, folder)
      #      copy_contents(c01_folder_path, dest_path)
       ##    check_permissions(dest_path)
            
        if folder in c1_folders:
            print(folder)
            folder=pad_filename(folder)
            print(folder)
            dest_path = os.path.join(c1_path, folder)
            copy_contents(c01_folder_path, dest_path)
            print("source-path:",c01_folder_path)
            print("destination-path",dest_path)
if __name__ == "__main__":
    # Path to the Subjects directory
    subjects_dir = r"C:\Users\A.C\Documents\Final year project (FYP)\Subjects"
    # Perform the operation
    copy_from_c01_to_c0_c1(subjects_dir)
