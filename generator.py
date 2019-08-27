import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
from skimage.transform import resize
from skimage.exposure import equalize_hist
from openface_tools import extract_action_units, stack_action_units, align_face
from mixup_generator import MixupGenerator
from saliency_map import compute_saliency


def normalize_illumination(img):
    img = rgb2hsv(img)
    value = img[:, :, 2]
    value = equalize_hist(value)
    img[:, :, 2] = value
    return hsv2rgb(img)


def random_grayscale(p=0.5):
    """ Returns a function that transforms a rgb image with probability p into a grayscale image with three channels. """

    def rndgray(image):
        rnd = np.random.rand()
        if rnd <= p:
            image = rgb2gray(image)
            image = np.expand_dims(image, axis=-1)
            image = np.tile(image, (1, 1, 3))
        return image

    return rndgray


def mixup(images, labels, alpha=0.2):
    gen = MixupGenerator(images, labels, batch_size=len(images), alpha=alpha)
    rnd = np.random.randint(0, gen.__len__())
    images, labels = gen.__getitem__(rnd)
    return images, labels


def crop_and_stack(image):
    """ Crops the image the same way as described in the work of Wang et al. (https://arxiv.org/abs/1905.04075) and stacks the crops up in the last axis. """
    assert image.shape[0] == image.shape[1], 'height and width of image must match'
    size = int(image.shape[0])
    crops = []
    crops.append(image[:int(0.75 * size), :int(0.75 * size), :])  # upper left crop (0.75 ratio)
    crops.append(image[:int(0.75 * size), int(0.25 * size):, :])  # upper right crop (0.75 ratio)
    crops.append(image[int(0.25 * size):, int(0.125 * size):int(0.875 * size), :])  # bottom center crop (0.75 ratio)
    crops.append(
        image[int(0.05 * size):int(0.95 * size), int(0.05 * size):int(0.95 * size), :])  # center crop (0.90 ratio)
    crops.append(
        image[int(0.075 * size):int(0.925 * size), int(0.075 * size):int(0.925 * size), :])  # center crop (0.85 ratio)
    for i in range(len(crops)):
        crops[i] = resize(crops[i], (size, size))
        image = np.concatenate([image, crops[i]], axis=2)
    return image


def extract_from_channels(image):
    """ Extracts the different crops that were generated using crop_and_stack. """
    num_channels = image.shape[2]
    assert num_channels == 6 or num_channels == 18
    if num_channels == 6:
        mode = 'grayscale'
    else:
        mode = 'color'
    channels_per_image = 3 if mode == 'color' else 1
    extracted = [image[:, :, channels_per_image * i:channels_per_image * (i + 1)] for i in range(7)]
    return extracted


def apply_saliency_map(image):
    """ Applies a saliency map to a given image. """
    map = compute_saliency(image)
    map = np.expand_dims(map, axis=-1)
    image = image * map
    image *= (1 / np.max(image))
    return image


def shuffle_arrays(*args):
    """
        Shuffles one or more arrays in the exact same way.
    """
    permutation = np.random.permutation(len(args[0]))
    shuffled = []
    for array in args:
        shuffled.append(array[permutation])
    return shuffled


def convert_to_one_hot(label):
    vec = np.zeros(8)
    vec[label] = 1
    return vec


def padding(image, output_size):
    assert image.shape[0] == image.shape[1]
    assert output_size > image.shape[0]
    diff = int(output_size - image.shape[0])
    p1 = diff // 2
    p2 = diff - p1
    out = np.zeros((output_size, output_size, 3))
    out[p1:-p2, p1:-p2, :] += image
    return out


class TrainDataGenerator(keras.utils.Sequence):
    """
        Generates uniform training samples.
    """

    # Generator specific properties
    DATA_DIR = '/data/cvg/luca/AffectNet/manually_annotated/preprocessed/train/emotion_only'
    MAX_LABEL_INDICES = {
        0: 63160,
        1: 113420,
        2: 21468,
        3: 11913,
        4: 5368,
        5: 3654,
        6: 23969,
        7: 3609,
    }
    MAX_INDEX = MAX_LABEL_INDICES[1]

    def __init__(self,
                 batch_size=64,
                 imsize=192,
                 pad_to_size=192,
                 num_samples='all',
                 augment=True,
                 mixup=False,
                 rndgray=False,
                 preprocess_func=None,
                 multi_crop=False,
                 hist_eq=True,
                 grayscale=False,
                 range255=False,
                 align=False,
                 mask=False,
                 saliency_map=False,
                 action_units=False,
                 shuffle=True):
        # Assertions
        assert batch_size % 8 == 0, 'Batch size should be a multiple of 8.'
        assert not (multi_crop and action_units), 'Cannot set multi_crop and action_units at the same time.'
        assert not (mask and not align), 'Mask can only be extracted if align is set to true.'
        assert num_samples == 'all' or (
                0 <= num_samples <= 8 * (self.MAX_INDEX + 1)), 'num_samples must be between 0 and {}'.format(
            8 * (self.MAX_INDEX + 1))
        assert not (imsize != 192 and pad_to_size != 192), 'You can only specify imsize or pad_to_size but not both'

        # Initialization
        self.CURRENT_LABEL_INDICES = {i: 0 for i in range(8)}
        self.NUM_DATA = sum(self.MAX_LABEL_INDICES.values())
        self.batch_size = batch_size
        self.imsize = imsize
        self.pad_to_size = pad_to_size
        self.fraction = 1.0 if num_samples == 'all' else num_samples / (8 * (self.MAX_INDEX + 1))
        self.augment = augment
        self.mixup = mixup
        self.preprocess_func = preprocess_func
        self.multi_crop = multi_crop
        self.hist_eq = hist_eq
        self.grayscale = grayscale
        self.range255 = range255
        self.align = align
        self.mask = mask
        self.saliency_map = saliency_map
        self.action_units = action_units
        self.shuffle = shuffle
        self.permutations = [np.random.permutation(self.MAX_LABEL_INDICES[lb] + 1) for lb in
                             range(len(self.MAX_LABEL_INDICES))]

        if rndgray:
            func = random_grayscale(p=0.5)
        else:
            func = None

        self.augmenter = ImageDataGenerator(rotation_range=25,
                                            width_shift_range=0.1,
                                            height_shift_range=0.1,
                                            shear_range=0.01,
                                            zoom_range=[0.9, 1.15],
                                            horizontal_flip=True,
                                            vertical_flip=False,
                                            fill_mode='reflect',
                                            data_format='channels_last',
                                            brightness_range=[0.5, 1.5],
                                            preprocessing_function=func)

    def __len__(self):
        """
            Denotes the number of batches per epoch
        """
        return int(self.fraction * (self.MAX_INDEX + 1) * 8 / self.batch_size)

    def __getitem__(self, index):
        """
            Generates one batch of data.
        """
        samples_per_category = int(self.batch_size / 8)

        data = []
        labels = []

        for label in range(len(self.MAX_LABEL_INDICES)):
            for i in range(samples_per_category):
                # Oversampling of all categories except MAX_INDEX
                label_permutation_idx = self.permutations[label][
                    (samples_per_category * index + i) % (self.MAX_LABEL_INDICES[label] + 1)]
                sample = np.load(
                    '{}/{}/{}_{}.npz'.format(self.DATA_DIR, label, label, '{0:0>6}'.format(label_permutation_idx)))
                sample = sample.f.arr_0
                if self.imsize != 192:
                    sample = resize(sample, (self.imsize, self.imsize))
                elif self.pad_to_size != 192:
                    sample = padding(sample, self.pad_to_size)
                if self.hist_eq:
                    sample = normalize_illumination(sample)
                if self.align:
                    sample = align_face(sample, mask=self.mask)
                if self.saliency_map:
                    sample = apply_saliency_map(sample)
                if self.grayscale:
                    sample = np.expand_dims(rgb2gray(sample), axis=-1)
                if self.range255:
                    sample = sample * 255
                if self.multi_crop:
                    sample = crop_and_stack(sample)
                elif self.action_units:
                    sample = stack_action_units(sample)
                category = convert_to_one_hot(label)
                data.append(sample)
                labels.append(category)

        data = np.array(data)
        labels = np.array(labels)

        if self.augment:
            if self.range255:
                aug = self.augmenter.flow(data, labels, batch_size=self.batch_size)
                data, labels = next(aug)
                if self.mixup:
                    data, labels = mixup(data, labels, alpha=0.2)
            else:
                aug = self.augmenter.flow(data * 255, labels, batch_size=self.batch_size)
                data, labels = next(aug)
                if self.mixup:
                    data, labels = mixup(data, labels, alpha=0.2)
                data = data / 255

        if not self.preprocess_func is None:
            for idx in range(len(data)):
                data[idx] = self.preprocess_func(data[idx])

        if self.shuffle:
            data, labels = shuffle_arrays(data, labels)

        return data, labels


class ValDataGenerator(TrainDataGenerator):
    """
        Generates uniform validation samples.
    """
    # Generator specific properties
    DATA_DIR = '/data/cvg/luca/AffectNet/manually_annotated/preprocessed/test/emotion_only'
    MAX_LABEL_INDICES = {
        0: 499,
        1: 499,
        2: 499,
        3: 499,
        4: 499,
        5: 499,
        6: 499,
        7: 499,
    }
    MAX_INDEX = MAX_LABEL_INDICES[1]
