import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from inference_nii_BCH_0814 import generate_dmap, normalize_intensity, process_single
from scipy.ndimage import zoom

def process_single(img, size, dmap=None, label=None):
    H, W = img.shape
    # crop the image to make it multiples of 16
    # Resize function for MRI (uses interpolation)
    def resize_mri(image, target_shape):
        zoom_factors = np.array(target_shape) / np.array(image.shape)
        return zoom(image, zoom_factors, order=3)  # Cubic interpolation for smooth results

    # Resize function for labels (uses nearest-neighbor to keep original values)
    def resize_label(label, target_shape):
        zoom_factors = np.array(target_shape) / np.array(label.shape)
        return zoom(label, zoom_factors, order=0)  # Nearest-neighbor to prevent new label values
    img_rsz = resize_mri(img, size)
    img_rsz = np.rot90(img_rsz, k=1)
    if label is not None:
        label_rsz = resize_label(label, size)
        label_rsz = np.rot90(label_rsz, k=1)
    if dmap is not None:
        dmap_rsz = resize_mri(dmap, size)
    if label is not None:
        return img_rsz, dmap_rsz, label_rsz
    else:
        return img_rsz, dmap_rsz

# Path to the dataset directory (modify this)
dataset_path = "./data/nifti/etv167_NIFTI_pre_reviewed_0219/"  # Change to your dataset folder
output_path = "./data/npy/etv167_NIFTI_pre_reviewed_0219/"
# Get a list of patient folders
patient_folders = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])

print("Processing NIFTI files and saving as .npy...")

# Iterate through each patient folder
for patient in tqdm(patient_folders):
    patient_path = os.path.join(dataset_path, patient)
    
    # Define MRI and label file paths
    mri_path = [x for x in os.listdir(patient_path) if 'seg' not in x][0]
    label_path = mri_path.split('.')[0] + '_seg.nii.gz'

    # Define corresponding output .npy file paths
    if not os.path.exists(os.path.join(output_path, patient)):
        os.makedirs(os.path.join(output_path, patient))
    save_npy_path = os.path.join(output_path, patient, mri_path.split('.')[0])
    
    # Load MRI and label data
    mri_nifti = nib.load(os.path.join(patient_path, mri_path))
    label_nifti = nib.load(os.path.join(patient_path, label_path))
    res = mri_nifti.header.get_zooms() #resolution
    
    # Convert to NumPy arrays
    mri_array = mri_nifti.get_fdata(dtype=np.float32)  # Convert to float32
    label_array = label_nifti.get_fdata()  # Convert to uint8 (segmentation masks)

    # Normalize MRI images (optional)
    mri_array = normalize_intensity(mri_array)
    dmap = generate_dmap(mri_array.shape[0], mri_array.shape[1], res[0], res[1])/255
    # Process single image and label (cropping, rotating, resizing)
    for i in range(mri_array.shape[2]):
        input, dmap, label = process_single(img=mri_array[:, :, i], size=(448, 448), dmap=dmap, label=label_array[:,:,i])
        label[(label != 0) & (label!=1) & (label!=2)] = 0
        np.save(save_npy_path + f'_{i}.npy', input)
        np.save(save_npy_path + f'_{i}_dmap.npy', dmap)
        np.save(save_npy_path + f'_{i}_seg.npy', label)

print("All files saved as .npy while preserving original structure!")
