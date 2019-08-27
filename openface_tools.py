import os
from datetime import datetime
from skimage.io import imread, imsave
import numpy as np
import pandas as pd

OPEN_FACE_BINARY_PATH = '/data/cvg/luca/tools/OpenFace/build/bin'
ALLOWED_ACTION_UNITS = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]


def extract_action_units(np_img, au_list=None):
    """
        Params:
            - np_img: numpy array of shape (height, width, num_channels)
                      containing the image
            - au_list: list of action units to extract. If not specified,
                       all action units listed above will be extracted.
        Returns:
            - numpy array containing the extracted action units
    """
    # Make sure, only extractable action units are passed as parameter
    if not au_list is None:
        assert all(au in ALLOWED_ACTION_UNITS for au in au_list), 'Invalid action unit list provided.'

    # Prepare extraction
    current_dir = os.getcwd()
    os.chdir(OPEN_FACE_BINARY_PATH)
    timestamp = datetime.now().timestamp()

    # Extract action units using OpenFace
    np_img *= 255
    np_img = np_img.astype(np.uint8)
    imsave('{}.jpg'.format(timestamp), np_img)
    exit_code = os.system('./FaceLandmarkImg -f {}.jpg >/dev/null'.format(timestamp))
    csv_data = pd.read_csv('processed/{}.csv'.format(timestamp))
    au_presence = csv_data.iloc[0, -18:].tolist()
    del au_presence[-2]  # remove AU_28c because it has no intensity equivalent
    au_intensity = csv_data.iloc[0, -35:-18].tolist()

    # Convert action unit lists to numpy arrays
    au_presence = np.array(au_presence)
    au_intensity = np.array(au_intensity)

    # Filter only wanted action units
    if not au_list is None:
        wanted_au_indices = [ALLOWED_ACTION_UNITS.index(au) for au in au_list]
        au_presence = au_presence[wanted_au_indices]
        au_intensity = au_intensity[wanted_au_indices]

    # Delete temporary files created during action unit extract_action_units
    exit_code = os.system('rm -r {}.jpg >/dev/null'.format(timestamp))
    exit_code = os.system('rm -r processed/{}* >/dev/null'.format(timestamp))

    os.chdir(current_dir)

    return au_presence, au_intensity


def stack_action_units(image):
    """
        Adds one channel to the image containing the 17 action unit intensities
        that OpenFace detects. The 17 intensity values are spread evenly spaced
        throughout the channel.
    """
    size = image.shape[0]
    _, au_intensities = extract_action_units(image)
    au_channel = np.zeros((size, size, 1))
    indices = _get_au_embedding_indices(image.shape)
    for au, rowcol in enumerate(indices):
        row, col = rowcol
        au_channel[row, col] = au_intensities[au]
    image = np.concatenate([image, au_channel], axis=-1)
    return image


def unstack_action_units(image):
    """
        Unstacks action units from the fourth channel embedding and returns
        the original image with three channels as well as a vector containing
        the 17 action unit intensities.
    """
    indices = _get_au_embedding_indices(image.shape)
    au_channel = image[:, :, -1]
    image = image[:, :, :4]
    au_intensities = np.zeros(17)
    for au, rowcol in enumerate(indices):
        row, col = rowcol
        au_intensities[au] += au_channel[row, col]
    return image, au_intensities


def _get_au_embedding_indices(shape):
    assert shape[0] == shape[1]
    size = shape[0]
    spacing = int(size ** 2 / 17)
    offset = int((size ** 2 - spacing * 17) / 2)
    indices = []
    for i in range(1, 17):
        location = i * spacing + offset
        col = int(location / size)
        row = np.mod(location, size)
        indices.append((row, col))
    return indices


def extract_pose(np_img):
    """Extracts pitch, yaw and roll from a face image.

    Args:
    np_img -- numpy array of shape (height, width, num_channels) containing the image.

    Returns:
    pose_params -- numpy array containing the extracted pose parameters: [Rx, Ry, Rz]

    """

    # Prepare extraction
    current_dir = os.getcwd()
    os.chdir(OPEN_FACE_BINARY_PATH)
    timestamp = datetime.now().timestamp()

    # Extract action units using OpenFace
    np_img *= 255
    np_img = np_img.astype(np.uint8)
    imsave('{}.jpg'.format(timestamp), np_img)
    exit_code = os.system('./FaceLandmarkImg -f {}.jpg >/dev/null'.format(timestamp))
    csv_data = pd.read_csv('processed/{}.csv'.format(timestamp), sep=',\s', engine='python')
    pose_params = csv_data[['pose_Rx', 'pose_Ry', 'pose_Rz']].iloc[0].tolist()

    # Convert pose parameters to numpy arrays
    pose_params = np.array(pose_params)

    # Delete temporary files created during action unit extract_action_units
    exit_code = os.system('rm -r {}.jpg >/dev/null'.format(timestamp))
    exit_code = os.system('rm -r processed/{}* >/dev/null'.format(timestamp))

    os.chdir(current_dir)

    return pose_params


def align_face(np_img, mask=True):
    """Uses OpenFace to align the face.

    Args:
    np_img -- numpy array of shape (height, width, num_channels) containing the image.
    mask -- if True, a mask is added to the face resulting in a black background.

    """
    # Prepare extraction
    current_dir = os.getcwd()
    os.chdir(OPEN_FACE_BINARY_PATH)
    timestamp = datetime.now().timestamp()
    mask_param = '-nomask' if mask is False else ''

    # Align and mask face using OpenFace
    np_img *= 255
    np_img = np_img.astype(np.uint8)
    imsave('{}.jpg'.format(timestamp), np_img)
    exit_code = os.system(
        './FaceLandmarkImg -f {}.jpg -wild -simalign -simsize 192 {} -format_aligned jpg >/dev/null'.format(timestamp,
                                                                                                            mask_param))
    aligned_img = imread('processed/{}_aligned/face_det_000000.jpg'.format(timestamp))
    aligned_img = aligned_img / 255.

    # Delete temporary files created during face alignment
    exit_code = os.system('rm -r {}.jpg >/dev/null'.format(timestamp))
    exit_code = os.system('rm -r processed/{}* >/dev/null'.format(timestamp))

    os.chdir(current_dir)

    return aligned_img
