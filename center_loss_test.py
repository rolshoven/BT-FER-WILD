from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Layer, Input, Dense, Flatten, BatchNormalization
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import losses
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import initializers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.regularizers import l2

from holonet import get_holonet
from lightcnn import low_level_edge_detector, dsrc_module, classification_module
from generator import TrainDataGenerator, ValidationDataGenerator
from tensorflow.python import keras
from tensorflow.python.keras.models import load_model

import os, sys
import numpy as np

### parameters

initial_learning_rate = 1e-3
batch_size = 64
epochs = 50
weight_decay = 0.0005


class CenterLossTrainGen(TrainDataGenerator):

    def __init__(self,
                 batch_size=64,
                 imsize=192,
                 fraction=1.0,
                 hist_eq=False,
                 rgb255=False,
                 shuffle=True):
        assert batch_size % 8 == 0, 'Batch size should be a multiple of 8.'
        assert 0 <= fraction <= 1.0, 'Fraction parameter must be between 0 and 1.0'
        super().__init__(batch_size=batch_size, imsize=imsize, fraction=fraction, hist_eq=hist_eq, rgb255=rgb255,
                         shuffle=shuffle)

    def __getitem__(self, index):
        """
            Generates one batch of data.
        """
        X_train, Y_train = super().__getitem__(index)

        dummy = np.zeros((X_train.shape[0], 1))

        wrapperX = [X_train, Y_train]
        wrapperY = [Y_train, dummy]

        return wrapperX, wrapperY


class CenterLossValGen(ValidationDataGenerator):

    def __init__(self,
                 batch_size=64,
                 imsize=192,
                 fraction=1.0,
                 hist_eq=False,
                 rgb255=False,
                 shuffle=True):
        assert batch_size % 8 == 0, 'Batch size should be a multiple of 8.'
        assert 0 <= fraction <= 1.0, 'Fraction parameter must be between 0 and 1.0'
        super().__init__(batch_size=batch_size, imsize=imsize, fraction=fraction, hist_eq=hist_eq, rgb255=rgb255,
                         shuffle=shuffle)

    def __getitem__(self, index):
        """
            Generates one batch of data.
        """
        X_train, Y_train = super().__getitem__(index)

        dummy = np.zeros((Y_train.shape[0], 1))

        wrapperX = [X_train, Y_train]
        wrapperY = [Y_train, dummy]

        return wrapperX, wrapperY


class CenterLossLayer(Layer):
    """ Center Loss implementation provided by Han Dongfeng (https://github.com/handongfeng/MNIST-center-loss) """

    def __init__(self, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(8, 8),
                                       initializer='uniform',
                                       trainable=False)
        # self.counter = self.add_weight(name='counter',
        #                                shape=(1,),
        #                                initializer='zeros',
        #                                trainable=False)  # just for debugging
        super().build(input_shape)

    def call(self, x, mask=None):
        # print(x[0].shape)
        # print(x[1].shape)

        # x[0] is Nx2, x[1] is Nx8 onehot, self.centers is 8x2
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 8x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 8x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)

        # self.add_update((self.counter, self.counter + 1), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True)  # / K.dot(x[1], center_counts)
        return self.result  # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


### custom loss

def zero_loss(y_true, y_pred):
    return 0.5 * K.sum(y_pred, axis=0)


### model

def get_model(lambda_centerloss, input_shape=(192, 192, 3), num_classes=8, regularizer=None, lr=0.01):
    image = Input(shape=input_shape)
    label = Input(shape=(num_classes,))
    X = low_level_edge_detector(image)
    X = dsrc_module(X, num_filters=16, module_name='a')
    X = dsrc_module(X, num_filters=32, module_name='b')
    X = dsrc_module(X, num_filters=64, module_name='c')
    X = dsrc_module(X, num_filters=128, module_name='d')
    X = dsrc_module(X, num_filters=256, module_name='e')
    X = dsrc_module(X, num_filters=512, module_name='f')
    X = Conv2D(filters=num_classes, kernel_size=(1, 1))(X)
    X = GlobalAveragePooling2D()(X)
    X = Flatten()(X)
    center_loss = CenterLossLayer(alpha=0.5, name='centerlosslayer')([X, label])
    if regularizer is None:
        X = Dense(num_classes, name='fc' + str(num_classes))(X)
    else:
        X = Dense(num_classes, name='fc' + str(num_classes), kernel_regularizer=l2(regularizer))(X)
    predicted_class = Activation('softmax', name='prediction')(X)

    model = Model(inputs=[image, label], outputs=[predicted_class, center_loss])
    model.compile(optimizer=optimizers.Adam(lr=lr),
                  loss=[losses.categorical_crossentropy, zero_loss],
                  loss_weights=[1, lambda_centerloss],
                  metrics=['accuracy'])

    return model


### run model

def train(lambda_centerloss):
    BATCH_SIZE = 32
    NUM_EPOCHS = 150

    MODEL_NAME = 'LightCNN_CenterLoss'

    model = get_model(lambda_centerloss)

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs/{}/'.format(MODEL_NAME), histogram_freq=0,
                                             write_graph=True, write_images=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_prediction_loss', factor=0.9, patience=3, verbose=0)
    checkpoint = keras.callbacks.ModelCheckpoint(
        '/data/cvg/luca/checkpoints/' + MODEL_NAME + '/weights-improvement-{epoch:02d}-{val_prediction_acc:.2f}.hdf5',
        monitor='val_prediction_loss',
        verbose=0,
        save_best_only=True)
    callbacks = [reduce_lr, tbCallBack, checkpoint]

    train_data_generator = CenterLossTrainGen(batch_size=BATCH_SIZE, fraction=0.01, hist_eq=True)  # ~9k samples
    validation_data_generator = CenterLossValGen(batch_size=BATCH_SIZE, hist_eq=True)

    model.fit_generator(generator=train_data_generator,
                        validation_data=validation_data_generator,
                        epochs=NUM_EPOCHS,
                        use_multiprocessing=True,
                        workers=6,
                        verbose=1,
                        callbacks=callbacks)

    K.clear_session()
    return


###

if __name__ == '__main__':
    train(0.1)
