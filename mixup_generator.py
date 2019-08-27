import numpy as np
from tensorflow.python import keras


class MixupGenerator(keras.utils.Sequence):
    """ Generates mixed up data for Keras, adapted implementation of https://github.com/yu4u/mixup-generator """

    def __init__(self, X_train, y_train, batch_size=32, shuffle=True, alpha=.2, datagen=None):
        'Initialization'
        self.batch_size = batch_size
        self.X_train = X_train
        self.y_train = y_train
        self.shuffle = shuffle
        self.on_epoch_end()
        self.alpha = alpha
        self.datagen = datagen

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X_train) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_train))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids]
        X2 = self.X_train[np.flip(batch_ids)]  # replaced this with flip
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        y1 = self.y_train[batch_ids]
        y2 = self.y_train[np.flip(batch_ids)]
        y = y1 * y_l + y2 * (1 - y_l)  # removed the list option

        return X / 255, y  # Rex added dividing by 255 here
