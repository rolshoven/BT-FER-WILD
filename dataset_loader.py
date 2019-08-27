import os
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2grey
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class CKPlus:
    DATA_ROOT = '/data/cvg/luca/CK+/cohn-kanade-images/'
    LABEL_ROOT = '/data/cvg/luca/CK+/Emotion/'
    PREPROCESSED_ROOT = '/data/cvg/luca/CK+/preprocessed/'

    # Labels used in CK+ dataset
    EMOTIONS = {
        0: 'Neutral',
        1: 'Anger',
        2: 'Contempt',
        3: 'Disgust',
        4: 'Fear',
        5: 'Happiness',
        6: 'Sadness',
        7: 'Surprise'
    }

    # Mapping from CK+ labels to our labels
    LABEL_TRANS = {
        0: 0,
        1: 6,
        2: 7,
        3: 5,
        4: 4,
        5: 1,
        6: 2,
        7: 3
    }

    def _get_identifiers(self):
        """
            Returns a list of file names that correspond to images having a label.
        """
        labels = []
        for _, _, f in os.walk(self.LABEL_ROOT):
            for file in f:
                labels.append(f[0][:-12])
        return labels

    def _convert_to_one_hot(self, label):
        vec = np.zeros(8)
        vec[label] = 1
        return vec

    def _extract_label(self, path):
        with open(path, 'r') as img:
            return self._convert_to_one_hot(int(img.readline().strip()[0]))

    def _convert_labels(self, labels):
        for i in range(labels.shape[0]):
            lb = np.where(labels == 1)[0][0]
            labels[i] = self._convert_to_one_hot(self.LABEL_TRANS[lb])
        return labels

    def load_sample(self, identifier, dims=None):
        """
            Loads a sample consisting of an image and the corresponding labels

            Params:
                - identifier:   an identifier string for a certain image,
                                e.g. 'S005_001_000000111'
                - dims:         dimension that the preprocessed image should
                                have of the form (height, width)
            Returns:
                image, label:   numpy arrays containing the image of the form
                                (dims[0], dims[1], 1) and the label of the form
                                (8,)
        """
        path = identifier.replace('_', '/')[:-8]
        img = imread('{}{}{}.png'.format(self.DATA_ROOT, path, identifier))
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        else:
            img = np.expand_dims(rgb2grey(img), axis=-1)
        if not dims is None:
            img = resize(img, dims)
        label = self._extract_label('{}{}{}_emotion.txt'.format(self.LABEL_ROOT, path, identifier))
        return img, label

    def load_all(self, dims=None):
        data = np.array([]).reshape(0, dims[0], dims[1], 1)
        labels = np.array([]).reshape(0, 8)
        for id in self._get_identifiers():
            img, lb = self.load_sample(id, (dims[0], dims[1]))
            data = np.vstack([data, np.array([img])])
            labels = np.vstack([labels, np.array([lb])])
        return data, self._convert_labels(labels)


class FER2013:
    DATA_CSV = '/data/cvg/luca/FER2013/fer2013.csv'
    PREPROCESSED_ROOT = '/data/cvg/luca/FER2013/preprocessed/'

    WIDTH = 48
    HEIGHT = 48

    EMOTIONS = {
        0: 'Anger',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happiness',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }

    LABEL_TRANS = {
        0: 6,
        1: 3,
        2: 4,
        3: 5,
        4: 2,
        5: 1,
        6: 0
    }

    def __init__(self, populate=False):
        self.X_train = self.Y_train = self.X_val = self.Y_val = self.X_test = self.Y_test = None
        if populate:
            data, labels = self.load_all()
            self.generate_sets(data, labels)

    def generate_sets(self, data, labels):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(data, labels, test_size=0.1,
                                                                                random_state=42)
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train, test_size=0.1,
                                                                              random_state=41)
        return (self.X_train, self.Y_train), (self.X_val, self.Y_val), (self.X_test, self.Y_test)

    def load_all(self):
        data = pd.read_csv(self.DATA_CSV)
        pixels = data['pixels'].tolist()
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(self.WIDTH, self.HEIGHT, 1)
            face = face / 255.0
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        labels = pd.get_dummies(data['emotion']).as_matrix()
        return faces, labels

    def get_train_data(self):
        assert self.X_train is not None and self.Y_train is not None
        return self.X_train, self.Y_train

    def get_validation_data(self):
        assert self.X_val is not None and self.Y_val is not None
        return self.X_val, self.Y_val

    def get_test_data(self):
        assert self.X_test is not None and self.Y_test is not None
        return self.X_test, self.Y_test
