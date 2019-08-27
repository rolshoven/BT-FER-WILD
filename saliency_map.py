from skimage.io import imread, imsave
from datetime import datetime
import numpy as np
import os

VENV_EXECUTABLE_DIR = '/data/cvg/luca/.envs/env27/bin/python'
MLNET_DIR = '/data/cvg/luca/tools/mlnet'
INPUT_DIR = 'faces'
OUTPUT_DIR = 'out'


def compute_saliency(np_img):
    """ Computes the saliency map of an image using a pretrained ML-Net (see https://github.com/marcellacornia/mlnet) """
    # Convert image
    if np.max(np_img) <= 1:
        np_img = np_img * 255
        np_img = np_img.astype(np.uint8)

    # Save image to disk
    timestamp = datetime.now().timestamp()
    imsave('{}/{}/{}.jpg'.format(MLNET_DIR, INPUT_DIR, timestamp), np_img)

    # Execute MLNET and read result into numpy array
    exit_code = os.system('{} {}/main.py test {}/ >/dev/null 2>&1'.format(VENV_EXECUTABLE_DIR, MLNET_DIR, INPUT_DIR))
    map = imread('{}/{}/{}.jpg'.format(MLNET_DIR, OUTPUT_DIR, timestamp))
    map = map / 255.

    # Remove temporary files created during the process
    exit_code = os.system('rm {}/{}/*.jpg >/dev/null 2>&1'.format(MLNET_DIR, INPUT_DIR))
    exit_code = os.system('rm {}/{}/*.jpg >/dev/null 2>&1'.format(MLNET_DIR, OUTPUT_DIR))

    return map


def alt(np_img):
    # Save image to disk
    timestamp = datetime.now().timestamp()
    np_arr_path = '{}/{}/{}.npy'.format(MLNET_DIR, INPUT_DIR, timestamp)
    np.save(np_arr_path, np_img)

    # Execute MLNET and read result into numpy array
    exit_code = os.system('{} {}/compute_map.py {} >/dev/null'.format(VENV_EXECUTABLE_DIR, MLNET_DIR, np_arr_path))

    map = np.load(np_arr_path[:-4] + '_out.npy')

    # Remove temporary files created during the process
    exit_code = os.system('rm {} >/dev/null'.format(np_arr_path))
    exit_code = os.system('rm {}_out.npy >/dev/null'.format(np_arr_path[:-4]))

    return map
