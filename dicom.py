import os

import matplotlib
import pydicom
import numpy as np
import scipy
from matplotlib import pyplot as plt, animation

############################### Code Source : Programming Activities ##########################
def median_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the median sagittal plane of the CT image provided."""
    return img_dcm[:, :, img_dcm.shape[1] // 2]  # Why //2?


def median_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the median sagittal plane of the CT image provided."""
    return img_dcm[:, img_dcm.shape[2] // 2, :]


def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the maximum intensity projection on the sagittal orientation."""
    return np.max(img_dcm, axis=2)


def AIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the average intensity projection on the sagittal orientation."""
    return np.mean(img_dcm, axis=2)


def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the maximum intensity projection on the coronal orientation."""
    return np.max(img_dcm, axis=1)


def AIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the average intensity projection on the coronal orientation."""
    return np.mean(img_dcm, axis=1)


def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """Rotate the image on the axial plane."""
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)

###############################################################################################

def find_centroid(mask: np.ndarray) -> np.ndarray:
    """
    Find the centroid of a binary mask.

    Args:
        mask (numpy.ndarray): The binary mask.

    Returns:
        numpy.ndarray: The centroid coordinates.
    """
    idcs = np.where(mask == 1)  # Find the indices of the voxels in the mask
    centroid = np.stack([np.mean(idcs[0]), np.mean(idcs[1]), np.mean(idcs[2])])  # Calculate the centroid coordinates
    return centroid

def normalize_array(arr):
    """
    Normalize the given array by scaling its values to a range of 0 to 100.

    Args:
        arr (numpy.ndarray): The input array to be normalized.

    Returns:
        numpy.ndarray: The normalized array.
    """
    max_value = np.max(arr)
    normalized_arr = (arr / max_value) * 100
    return np.round(normalized_arr).astype(int)


def alpha_fusion(img: np.ndarray, mask: np.ndarray, mask_centroid: np.ndarray) -> np.ndarray:
    """
    Visualize the axial slice (first dimension) of a single region with alpha fusion.

    Args:
        img (numpy.ndarray): The image as a 3D numpy array.
        mask (numpy.ndarray): The mask as a 3D numpy array.
        mask_centroid (numpy.ndarray): The centroid coordinates of the mask region.

    Returns:
        numpy.ndarray: The fused slices as a 2D numpy array.

    """

    # Select the axial slice of the image
    img_slice = img[:, :, :]

    # Select the axial slice of the mask
    mask_slice = mask[:, :, :] 

    # Initialize an empty list to store the fused slices
    fused_slices = []

    # Iterate over each slice along the first dimension of the image
    for i in range(img.shape[0]):
        # Set the colormap for the image slice
        cmap = matplotlib.colormaps['bone']
        
        # Normalize the pixel values of the image slice to the range of the colormap
        norm = matplotlib.colors.Normalize(vmin=np.amin(img_slice[i]), vmax=np.amax(img_slice[i]))
        
        # Perform alpha fusion by blending the image slice and the mask slice
        fused_slice = \
            0.8 * cmap(norm(img_slice[i]))[..., :3] + \
            0.2 * np.stack([mask_slice[i], np.zeros_like(mask_slice[i]), np.zeros_like(mask_slice[i])], axis=-1)
        
        # Append the fused slice to the list of fused slices
        fused_slices.append(fused_slice[..., 0])
    
    # Convert the list of fused slices to a numpy array
    fused_slices = np.array(fused_slices)
    
    # Return the fused slices
    return fused_slices

if __name__ == "__main__":
    # Initialize empty lists to store the pixel data and the segmentation data
    pixel_data = []
    segmentation_data = []
    # Read the segmentation data
    segmentation_path = "./HCC_011/05-23-1998-NA-AP-LIVER-85429/300.000000-Segmentation-86044/1-1.dcm"
    segmentation_dataset = pydicom.dcmread(segmentation_path)
    segmentation_array = segmentation_dataset.pixel_array
    # Iterate over the directory containing the DICOM files
    directory = "./HCC_011/05-23-1998-NA-AP-LIVER-85429/4.000000-Recon 2 LIVER 3 PHASE AP-74786"
    directories = sorted(os.listdir(directory))
    # Iterate over the DICOM files
    for filename in directories:
        # If the file is a DICOM file
        if filename.endswith(".dcm"):
            # Read the DICOM file
            path = os.path.join(directory, filename)
            dataset = pydicom.dcmread(path)
            # If the slice is the one with AcquisitionsNumber = 2 then take the pixel data
            if dataset.AcquisitionNumber == 2:
                pixel_data.append(dataset.pixel_array)
    
    # Pixel length in mm
    pixel_len_mm = [5, 0.78, 0.78]
    
    # Convert the list of pixel data to a numpy array
    img_dcm = np.array(pixel_data)
    
    # Normalize the pixel values of the image
    img_dcm = normalize_array(img_dcm)

    # Flip the segmentation array along the first dimension
    segmentation_array = np.flip(segmentation_array, axis=1)

    # The tumor sequence is the 79th to 158th slices
    tumor_sequence = segmentation_array[79:158]

    mask_centroid = find_centroid(tumor_sequence) # Tumor sequence
    segmented_img_dcm = alpha_fusion(img_dcm, tumor_sequence, mask_centroid)

    # Create projections varying the angle of rotation on sagittal plane
    img_min_seg = np.amin(segmented_img_dcm)
    img_max_seg = np.amax(segmented_img_dcm)
    cm_seg = matplotlib.colormaps["bone"]

    fig, ax = plt.subplots()
    os.makedirs("mip_animation/", exist_ok=True)

    n = 16
    projections = []
    projections_seg = []

    for idx, alpha in enumerate(np.linspace(0, 360 * (n - 1) / n, num=n)):

        rotated_img_seg = rotate_on_axial_plane(segmented_img_dcm, alpha)
        projection_seg = MIP_sagittal_plane(rotated_img_seg)

        ax.imshow(
            projection_seg,
            cmap=cm_seg,
            vmin=img_min_seg,
            vmax=img_max_seg,
            aspect=pixel_len_mm[0] / pixel_len_mm[1],
        )

        plt.savefig(f"mip_animation/{idx}.png")
        projections_seg.append(projection_seg)

    animation_data = [
        [
            ax.imshow(
                img_seg,
                animated=True,
                cmap=cm_seg,
                vmin=img_min_seg,
                vmax=img_max_seg,
                aspect=pixel_len_mm[0] / pixel_len_mm[1],
            )
        ]
        for img_seg in projections_seg
    ]
    anim = animation.ArtistAnimation(fig, animation_data, interval=15, blit=True)
    anim.save("mip_animation/animation.gif")
    plt.show()