import os

from skimage import exposure
import matplotlib
import pydicom
import numpy as np
from scipy.optimize import least_squares
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
import math
import scipy
import glob


############################ Code Source : Programming Activities #############################
def visualize_axial_slice(
        img: np.ndarray,
        mask: np.ndarray,
        mask_centroid: np.ndarray,
        ):
    """ Visualize the axial slice (firs dim.) of a single region with alpha fusion. """
    img_slice = img[mask_centroid[0].astype('int'), :, :]
    mask_slice = mask[mask_centroid[0].astype('int'), :, :]

    cmap = matplotlib.colormaps["bone"]
    norm = matplotlib.colors.Normalize(vmin=np.amin(img_slice), vmax=np.amax(img_slice))
    fused_slice = \
        0.5*cmap(norm(img_slice))[..., :3] + \
        0.5*np.stack([mask_slice, np.zeros_like(mask_slice), np.zeros_like(mask_slice)], axis=-1)

    return fused_slice

def multiply_quaternions(
        q1: tuple[float, float, float, float],
        q2: tuple[float, float, float, float]
        ) -> tuple[float, float, float, float]:
    """ Multiply two quaternions, expressed as (1, i, j, k). """
    # Your code here:
    #   ...
    return (
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    )

def conjugate_quaternion(
        q: tuple[float, float, float, float]
        ) -> tuple[float, float, float, float]:
    """ Multiply two quaternions, expressed as (1, i, j, k). """
    # Your code here:
    #   ...
    return (
        q[0], -q[1], -q[2], -q[3]
    )

def translation(
        point: tuple[float, float, float],
        translation_vector: tuple[float, float, float]
        ) -> tuple[float, float, float]:
    """ Perform translation of `point` by `translation_vector`. """
    x, y, z = point
    v1, v2, v3 = translation_vector
    return (x+v1, y+v2, z+v3)

def axial_rotation(
        point: tuple[float, float, float],
        angle_in_rads: float,
        axis_of_rotation: tuple[float, float, float]) -> tuple[float, float, float]:
    """ Perform axial rotation of `point` around `axis_of_rotation` by `angle_in_rads`. """
    x, y, z = point
    v1, v2, v3 = axis_of_rotation
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v1, v2, v3]]))
    v1, v2, v3 = v1 / v_norm, v2 / v_norm, v3 / v_norm
    p = (0, x, y, z)
    #   Quaternion associated to axial rotation.
    cos, sin = math.cos(angle_in_rads / 2), math.sin(angle_in_rads / 2)
    q = (cos, sin * v1, sin * v2, sin * v3)
    #   Quaternion associated to image point
    q_star = conjugate_quaternion(q)
    p_prime = multiply_quaternions(q, multiply_quaternions(p, q_star))
    #   Interpret as 3D point (i.e. drop first coordinate)
    return p_prime[1], p_prime[2], p_prime[3]

def translation_then_axialrotation(point: tuple[float, float, float], parameters: tuple[float, ...]):
    """ Apply to `point` a translation followed by an axial rotation, both defined by `parameters`. """
    x, y, z = point
    t1, t2, t3, angle_in_rads, v1, v2, v3 = parameters
    t_x, t_y, t_z = translation([x, y, z], [t1, t2, t3])
    r_x, r_y, r_z = axial_rotation([t_x, t_y, t_z], angle_in_rads,[v1, v2, v3])
    return [r_x, r_y, r_z]

def vector_of_residuals(ref_points: np.ndarray, inp_points: np.ndarray) -> np.ndarray:
    """ Given arrays of 3D points with shape (point_idx, 3), compute vector of residuals as their respective distance """
    distances = np.linalg.norm(inp_points - ref_points, axis=1)
    return distances

def coregister_landmarks(ref_landmarks: np.ndarray, inp_landmarks: np.ndarray):
    """ Coregister two sets of landmarks using a rigid transformation. """
    initial_parameters = [
        0, 0, 0,    # Translation vector
        0,          # Angle in rads
        1, 0, 0,    # Axis of rotation
    ]
    # Find better initial parameters
    centroid_ref = np.mean(ref_landmarks, axis=0)
    centroid_inp = np.mean(inp_landmarks, axis=0)
    initial_parameters[0] = centroid_ref[0] - centroid_inp[0]
    initial_parameters[1] = centroid_ref[1] - centroid_inp[1]
    initial_parameters[2] = centroid_ref[2] - centroid_inp[2]

    def function_to_minimize(parameters):
        """ Transform input landmarks, then compare with reference landmarks."""
        new_inp_landmarks = []
        for point in inp_landmarks:
            new_point = translation_then_axialrotation(point, parameters)
            new_inp_landmarks.append(new_point)
        new_inp_landmarks = np.array(new_inp_landmarks)
        current_value = vector_of_residuals(ref_landmarks, new_inp_landmarks)
        return current_value
    

    # Apply least squares optimization
    result = least_squares(
        function_to_minimize,
        x0=initial_parameters,
        verbose=0)
    return result

def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """Rotate the image on the axial plane."""
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1, 2), reshape=False)

def rotate_on_axial_plane_rgb(img: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """Rotate the image on the axial plane."""
    rotated_img = np.zeros_like(img)
    rotated_img[:,:] = scipy.ndimage.rotate(img[:,:], angle_in_degrees, reshape=False)
    return rotated_img

def median_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the median sagittal plane of the CT image provided."""
    return img_dcm[:, img_dcm.shape[2] // 2, :]

def mean_absolute_error(img_input: np.ndarray, img_reference) -> np.ndarray:
    """ Compute the MAE between two images. """
    return np.mean(np.abs(img_input - img_reference))

def mean_squared_error(img_input: np.ndarray, img_reference) -> np.ndarray:
    """ Compute the MSE between two images. """
    return np.mean((img_input - img_reference)**2)

def mutual_information(img_input: np.ndarray, img_reference) -> np.ndarray:
    """ Compute the Shannon Mutual Information between two images. """
    nbins = [10, 10]
    # Compute entropy of each image
    hist = np.histogram(img_input.ravel(), bins=nbins[0])[0]
    prob_distr = hist / np.sum(hist)
    entropy_input = -np.sum(prob_distr * np.log2(prob_distr + 1e-7))  # Why +1e-7?
    hist = np.histogram(img_reference.ravel(), bins=nbins[0])[0]
    prob_distr = hist / np.sum(hist)
    entropy_reference = -np.sum(prob_distr * np.log2(prob_distr + 1e-7))  # Why +1e-7?
    # Compute joint entropy
    joint_hist = np.histogram2d(img_input.ravel(), img_reference.ravel(), bins=nbins)[0]
    prob_distr = joint_hist / np.sum(joint_hist)
    joint_entropy = -np.sum(prob_distr * np.log2(prob_distr + 1e-7))
    # Compute mutual information
    return entropy_input + entropy_reference - joint_entropy

def median_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """Compute the median sagittal plane of the CT image provided."""
    return img_dcm[:, :, img_dcm.shape[1] // 2]

###############################################################################################

def preprocess_landmarks(landmarks):
    """
    Preprocesses the landmarks by normalizing and scaling them.

    Args:
        landmarks (numpy.ndarray): Array of landmarks.

    Returns:
        numpy.ndarray: Preprocessed landmarks.
    """
    # Normalize the landmarks to the range [0, 1]
    normalized_landmarks = landmarks / np.max(landmarks)
    # Scale the normalized landmarks to the range [0, 100] and round to 2 decimal places
    preprocessed_landmarks = np.round(normalized_landmarks * 100, 2)
    return preprocessed_landmarks.astype(int)


def normalize_intensity(input_image, reference_image):
    """
    Normalizes the intensity of the input image to match the intensity distribution of the reference image.

    Args:
        input_image (numpy.ndarray): Input image to be normalized.
        reference_image (numpy.ndarray): Reference image for intensity matching.

    Returns:
        numpy.ndarray: Normalized image with intensity matched to the reference image.
    """
    # Flatten the images to 1D arrays
    input_flat = input_image.flatten()
    reference_flat = reference_image.flatten()
    # Perform histogram matching
    matched_flat = exposure.match_histograms(input_flat, reference_flat)
    # Reshape the matched image back to its original shape
    matched_image = np.reshape(matched_flat, input_image.shape)
    return matched_image

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
    
def get_thalamus_mask(img_atlas: np.ndarray) -> np.ndarray:
    """
    Generates a binary mask for the thalamus region in the input atlas image.

    Args:
        img_atlas (numpy.ndarray): Input atlas image.

    Returns:
        numpy.ndarray: Binary mask of the thalamus region.
    """
    result = np.zeros_like(img_atlas)  # Initialize an array of zeros with the same shape as the input image
    # Set the pixels within the range [121, 150] to 1 in the result array, indicating the thalamus region
    result[(121 <= img_atlas) & (img_atlas <= 150)] = 1
    return result

def get_brain_pixel_data():
    # Recursive file search pattern
    file_pattern = os.path.join("./coregistration/RM_Brain_3D-SPGR", "**/*.dcm")

    # Get list of files matching the pattern
    dcm_files = glob.glob(file_pattern, recursive=True)

    instances = {}

    # Collect instances and their corresponding paths
    for file_path in dcm_files:
        dataset = pydicom.dcmread(file_path)
        instance_number = dataset.InstanceNumber
        instances[instance_number] = file_path

    # Sort instances by instance number and extract sorted paths
    sorted_instances = sorted(instances.items(), key=lambda x: x[0])
    sorted_paths = [path for _, path in sorted_instances]

    pixel_data = []

    # Collect pixel data from sorted paths
    for path in sorted_paths:
        dataset = pydicom.dcmread(path)
        pixel_data.append(np.flip(dataset.pixel_array, axis=0))
    return pixel_data
#################################### Visualize the results ####################################
def visualize_images(images, plot_title=""):
    """
    Visualize the input, reference, and AAL images in a single row.

    Args:
        images (list): List of tuples in the form (image, title).
        plot_title (str): Title of the plot.
    """
    num_images = len(images)
    num_cols = num_images

    # Create a figure and axes to hold the subplots
    fig, ax = plt.subplots(1, num_cols)

    # Iterate over the images and titles
    for i, (image, title) in enumerate(images):
        ax[i].imshow(image, cmap=matplotlib.colormaps["bone"])
        ax[i].set_title(title)

    # Set the title for the figure
    plt.suptitle(plot_title)
    plt.tight_layout()
    plt.show()

    # Save the figure
    # fig.savefig(f"{plot_title}.png", dpi=300)
###############################################################################################


if __name__ == '__main__':
    # First, we load the reference data
    ref_dcm = pydicom.dcmread("./coregistration/icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm").pixel_array
    # Load in the AAL
    aal_dcm = pydicom.dcmread("./coregistration/AAL3_1mm.dcm").pixel_array

    # Volumes
    pixel_data = get_brain_pixel_data()
    input_volume = np.stack(pixel_data, axis=0)

    # Reference pixel array to 3D volume
    ref_volume = np.stack(ref_dcm, axis=0)

    # AAL pixel array to 3D volume
    aal_volume = np.stack(aal_dcm, axis=0)

    # Compute the index of the central plane for each volume
    input_central_plane = input_volume.shape[0] // 2
    ref_central_plane = ref_volume.shape[0] // 2
    aal_central_plane = aal_volume.shape[0] // 2

    print("Input volume shape:", input_volume.shape)
    print("Reference volume shape:", ref_volume.shape)
    print("AAL volume shape:", aal_volume.shape)

    print("Input central plane:", input_central_plane)
    print("Reference central plane:", ref_central_plane)
    print("AAL central plane:", aal_central_plane)

    # With 15 it looks better and more similar to the reference image
    horizontal_plane = input_volume[input_central_plane + 15, :, :]
    horizontal_plane_ref = ref_volume[ref_central_plane, :, :]
    horizontal_plane_aal = aal_volume[aal_central_plane, :, :]
    
    # Initial visualizations
    visualize_images([(horizontal_plane, "Input"), (horizontal_plane_ref, "Reference"), (horizontal_plane_aal, "AAL")], "Starting Images")

    # preprocess the input image and the reference image to match the AAL image
    img_phantom = ref_dcm[6:-6, 6:-7, 6:-6]     # Crop phantom to atlas size
    aal_volume = aal_volume[:, :-1, :]     # Crop atlas size so that the sum of the shapes is divisible by 3

    print("Phantom shape:", img_phantom.shape)
    print("AAL shape:", aal_volume.shape)

    # Create the thalamus mask based on the atlas
    th_mask = get_thalamus_mask(aal_volume)
    m_centroid = find_centroid(th_mask)
    m_centroid_idx = m_centroid[0].astype(int)

    # Reshape the input data (crop, zoom, and initial rotation)
    pixel_data = np.array(pixel_data)  # Convert pixel_data to a numpy array
    z_start = pixel_data.shape[0] - 181 + 3  # Start index for cropping, adjusted with the mask
    z_end = z_start + 181  # End index for cropping
    cropped_data = pixel_data[z_start:z_end, 48:456, 83:438]  # Crop the data to desired dimensions

    # Calculate resize factors based on cropped data shape
    resize_factors = (
        181 / cropped_data.shape[0],  # Resize factor for the first dimension
        216 / cropped_data.shape[1],  # Resize factor for the second dimension
        181 / cropped_data.shape[2]   # Resize factor for the third dimension
    )

    zoom_data = zoom(cropped_data, resize_factors, order=1)  # Zoom the cropped data using the resize factors

    rotate_val = 3  # Rotation value
    rotated_data = rotate_on_axial_plane(zoom_data, rotate_val)  # Rotate the zoomed data on the axial plane

    # Now all the images are at the same size and orientation
    processed_input = normalize_intensity(rotated_data, img_phantom)  # Normalize intensity of the rotated data
    processed_input = preprocess_landmarks(processed_input)  # Preprocess the normalized data

    visualize_images(
        [(processed_input[m_centroid_idx], "Input"), (img_phantom[m_centroid_idx], "Reference"), (aal_volume[m_centroid_idx], "AAL")], 
        "Preprocessed with thalamus"
    )
    visualize_images(
        [(processed_input[m_centroid_idx], "Input"), (img_phantom[m_centroid_idx], "Reference")],
        "Horizontal plane before coregistration"
    )
    visualize_images( [
            (median_sagittal_plane(np.flip(processed_input)), "Sagittal input"),
            (median_coronal_plane(np.flip(processed_input)), "Coronal input"),
            (median_sagittal_plane(np.flip(img_phantom)), "Sagittal reference"),
            (median_coronal_plane(np.flip(img_phantom)), "Coronal reference")
            ],"Comparison before coregistration"
    )

    # It takes a while to run coregistration so 
    # we downsample the images to speed up the process
    downsampling_rate = 4
    ds_ref =  img_phantom[::4, ::4, ::4].reshape(-1,3)
    ds_inp_shape = processed_input[::4, ::4, ::4].shape
    ds_inp = processed_input[::4, ::4, ::4].reshape(-1,3)

    # We create the landmarks for the reference image and the input image
    ref_lm = img_phantom.reshape(-1,3)
    inp_lm = processed_input.reshape(-1,3)

    # Let's see the metrics before coregistration
    print('Residual vector of distances between each pair of landmark points:')
    vec = vector_of_residuals(ref_lm, inp_lm)
    print("  >> Residual vector shape:", vec.shape)
    print(f'  >> Mean: {np.mean(vec.flatten())}.')
    print(f'  >> Max: {np.max(vec.flatten())}.')
    print(f'  >> Min: {np.min(vec.flatten())}.')
    print(f'  >> Result: {vec}.')

    # Lets visualize the landmarks before coregistration
    limit = inp_lm.shape[0]
    # Vector of residuals: visualization
    ref_show = ref_lm[:limit]
    inp_show = inp_lm[:limit]
    fig = plt.figure()
    axs = np.asarray([fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')])
    axs[0].scatter(ref_show[..., 0], ref_show[..., 1], ref_show[..., 2], marker='o')
    axs[0].set_title('Reference landmarks')
    axs[1].scatter(inp_show[..., 0], inp_show[..., 1], inp_show[..., 2], marker='^')
    axs[1].set_title('Input landmarks')
    # Uniform axis scaling
    all_points = np.concatenate([ref_show, inp_show], axis=0)
    range_x = np.asarray([np.min(all_points[..., 0]), np.max(all_points[..., 0])])
    range_y = np.asarray([np.min(all_points[..., 1]), np.max(all_points[..., 1])])
    range_z = np.asarray([np.min(all_points[..., 2]), np.max(all_points[..., 2])])
    max_midrange = max(range_x[1]-range_x[0], range_y[1]-range_y[0], range_z[1]-range_z[0]) / 2
    for ax in axs.flatten():
        ax.set_xlim3d(range_x[0]/2 + range_x[1]/2 - max_midrange, range_x[0]/2 + range_x[1]/2 + max_midrange)
        ax.set_ylim3d(range_y[0]/2 + range_y[1]/2 - max_midrange, range_y[0]/2 + range_y[1]/2 + max_midrange)
        ax.set_zlim3d(range_z[0]/2 + range_z[1]/2 - max_midrange, range_z[0]/2 + range_z[1]/2 + max_midrange)
    fig.suptitle("Landmark points comparison before coregistration")
    plt.show()
    
    # We run the coregistration algorithm
    # We use the downsampled images and landmarks
    coregistration = coregister_landmarks(ds_ref[:limit], ds_inp[:limit])
    optimal_params = coregistration.x
    print("Optimal parameters:", optimal_params)
    t1, t2, t3, angle_in_rads, v1, v2, v3 = optimal_params
    print('Best parameters:')
    print(f'  >> Translation: ({t1:0.02f}, {t2:0.02f}, {t3:0.02f}).')
    print(f'  >> Rotation: {angle_in_rads:0.02f} rads around axis ({v1:0.02f}, {v2:0.02f}, {v3:0.02f}).')

    inp_lm = np.asarray([translation_then_axialrotation(point, optimal_params) for point in inp_lm[:]])

    
    fig = plt.figure()
    axs = np.asarray([fig.add_subplot(121, projection='3d'), fig.add_subplot(122, projection='3d')])
    axs[0].scatter(ref_show[..., 0], ref_show[..., 1], ref_show[..., 2], marker='o')
    axs[0].set_title('Reference landmarks')
    axs[1].scatter(inp_show[..., 0], inp_show[..., 1], inp_show[..., 2], marker='^')
    axs[1].set_title('Input landmarks')
    # Uniform axis scaling
    all_points = np.concatenate([ref_show, inp_show], axis=0)
    range_x = np.asarray([np.min(all_points[..., 0]), np.max(all_points[..., 0])])
    range_y = np.asarray([np.min(all_points[..., 1]), np.max(all_points[..., 1])])
    range_z = np.asarray([np.min(all_points[..., 2]), np.max(all_points[..., 2])])
    max_midrange = max(range_x[1]-range_x[0], range_y[1]-range_y[0], range_z[1]-range_z[0]) / 2
    for ax in axs.flatten():
        ax.set_xlim3d(range_x[0]/2 + range_x[1]/2 - max_midrange, range_x[0]/2 + range_x[1]/2 + max_midrange)
        ax.set_ylim3d(range_y[0]/2 + range_y[1]/2 - max_midrange, range_y[0]/2 + range_y[1]/2 + max_midrange)
        ax.set_zlim3d(range_z[0]/2 + range_z[1]/2 - max_midrange, range_z[0]/2 + range_z[1]/2 + max_midrange)
    fig.suptitle("Landmark points comparison after coregistration")
    plt.show()

    inp_lm_original = inp_lm.reshape(181, 216, 181)

    # Sagittal and coronal comparison
    fig, ax = plt.subplots(2, 2, figsize=(6,7))
    images = [
        median_sagittal_plane(np.flip(inp_lm_original)),
        median_coronal_plane(np.flip(inp_lm_original)),
        median_sagittal_plane(np.flip(img_phantom)),
        median_coronal_plane(np.flip(img_phantom))
    ]
    titles = [
        "Sagittal input",
        "Coronal input",
        "Sagittal reference",
        "Coronal reference"
    ]
    for i, (image, title) in enumerate(zip(images, titles)):
        ax[i//2, i%2].imshow(image, cmap=matplotlib.colormaps["bone"])
        ax[i//2, i%2].set_title(title)

    fig.suptitle("Sagittal and Coronal median comparison after coregistration")
    plt.show()

    fig, ax = plt.subplots(1, 2)
    images = [inp_lm_original[m_centroid_idx], img_phantom[m_centroid_idx]]
    titles = ["Input image", "Reference image"]
    for i in range(2):
        ax[i].imshow(images[i], cmap=matplotlib.colormaps["bone"])
        ax[i].set_title(titles[i])
    fig.suptitle("Input image and Reference image in horizontal plane\n after coregistration")
    plt.show()

    volume = np.stack(inp_lm_original, axis=0)
    volume_ref = np.stack(img_phantom, axis=0)
    volume_aal = np.stack(aal_dcm, axis=0)
    plane_index = volume.shape[0] // 2 
    plane_index_ref = volume_ref.shape[0] // 2
    plane_index_aal = volume_aal.shape[0] // 2
    

    img_orig = inp_lm_original[m_centroid_idx, :, :]
    img_ref = img_phantom[m_centroid_idx, :, :]
    
    mae = mean_absolute_error(img_ref, img_orig)
    print('MAE:')
    print(f'  >> Result: {mae:.02f} HU')

    mse = mean_squared_error(img_ref, img_orig)
    print('MSE:')
    print(f'  >> Result: {mse:.02f} HU^2')

    mutual_inf = mutual_information(img_ref, img_orig)
    print('Mutual Information:')
    print(f'  >> Result: {mutual_inf:02f} bits')
    
    fused_phantom = visualize_axial_slice(img_phantom, th_mask, m_centroid)
    fused_orig = visualize_axial_slice(inp_lm_original, th_mask, m_centroid)

    visualize_images([(fused_orig, 'Input image'), (fused_phantom, 'Reference image')], "Input and Reference image fused with thalamus region")
    # Invert input image from reference space into input space
    # All values on the y axis -0.35, all values on x axis -0.34
    red_channel = fused_orig[..., 0]
    fused_flat = red_channel.flatten()
    original_flat = img_orig.flatten()
    # Match intensity with high intensity orig image
    denormalized_flat = exposure.match_histograms(fused_flat, original_flat)
    denormalized_image = np.reshape(denormalized_flat, red_channel.shape)
    # Translate back from coregistration best values for translation
    decoregistration_params = [1.78, 2.00, 2.38, -0.15, -0.62, -0.51, -0.62]
    pre_translate = denormalized_image.reshape(-1, 3)
    translated = np.array([translation_then_axialrotation(point, decoregistration_params) for point in pre_translate])
    translated = translated.reshape(red_channel.shape)
    # Normalize intensity with original image 
    denormalized_intensity = normalize_intensity(translated,  rotated_data[plane_index])
    # Rotate back from the input image preprocessing
    degrees = -3
    rotated = rotate_on_axial_plane_rgb(denormalized_intensity, degrees)
    # Calculate the resize factors in reverse
    resize_factors = (512 / rotated.shape[0]/1.2, 512 / rotated.shape[1]/1.4)
    # Resize the image back to its original size
    reverted_data = zoom(rotated, resize_factors, order=1)
    # Determine the desired padding sizes
    pad_height = (512 - reverted_data.shape[0]) // 2
    pad_width = (512 - reverted_data.shape[1]) // 2
    # Pad the image with zeros
    padded_image = np.pad(reverted_data, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    visualize_images([(fused_orig, 'Input image fused with thalamus region in reference space'), (padded_image, 'Input image fused with thalamus region in input space'), (pixel_data[m_centroid_idx + 33], 'Original Input image')], "Final results")