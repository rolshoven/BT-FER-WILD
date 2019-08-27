from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from layers import GlobalCovPooling2D
import tensorflow as tf
import numpy as np


def residual_connection(X):
    return tf.atan(X)


def bregnet_module(X, num_filters, block_name, module_name, alpha=1.0):
    main = Conv2D(filters=num_filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                  name='conv_{}_{}_1'.format(block_name, module_name))(X)
    main = ELU(alpha=alpha, name='elu_{}_{}_1'.format(block_name, module_name))(main)
    main = Conv2D(filters=num_filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                  name='conv_{}_{}_2'.format(block_name, module_name))(main)
    main = BatchNormalization(name='bn_{}_{}'.format(block_name, module_name))(main)
    if K.int_shape(X)[-1] != num_filters:
        X = Conv2D(filters=num_filters, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal',
                   name='conv_{}_{}_skip'.format(block_name, module_name))(X)
    shortcut = Lambda(residual_connection)(X)
    out = Add()([main, shortcut])
    return out


def bregnet_block(X, num_modules, num_filters, block_name, alpha=1.0, debug=False):
    """Constructs a block of bregnet modules.

    Keyword arguments:
    num_modules -- the number of modules that should be built.
    num_filters -- list containing the number of filters that are evenly spread
                   across the modules.

    Example:
    num_modules=10 and num_filters=[32, 64] would result in 5 bregnet modules with
    32 filters and 5 modules with 64 filters.
    """
    filters = np.ones(num_modules)
    if len(filters) == 1:
        filters = filters * num_filters[0]
    else:
        filters = np.array(np.array_split(filters, len(num_filters)))
        if filters.dtype == 'O':
            # If num_modules % num_filters != 0, broadcasting works different
            filters *= np.array(num_filters)
        else:
            filters *= np.array(num_filters).reshape(-1, 1)
        filters = np.concatenate(filters).astype(np.int32)
    for idx in range(num_modules):
        X = bregnet_module(X, num_filters=filters[idx], block_name=block_name, module_name=idx + 1, alpha=alpha)
    return X


def downsampling(X, num_filters, block_name, alpha=1.0):
    main = Conv2D(filters=num_filters, kernel_size=(3, 3), strides=(2, 2), padding='same',
                  kernel_initializer='he_normal', name='conv_downsampling_{}'.format(block_name))(X)
    main = ELU(alpha=alpha, name='elu_downsampling_{}_1'.format(block_name))(main)
    main = Conv2D(filters=num_filters, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal',
                  name='conv_downsampling_{}_no_reduction'.format(block_name))(main)
    main = BatchNormalization(name='bn_downsampling_{}'.format(block_name))(main)
    shortcut = Lambda(residual_connection)(X)
    shortcut = AveragePooling2D(pool_size=(2, 2), padding='same')(shortcut)
    out = Add()([main, shortcut])
    return out


def BREGNet(input_shape=(192, 192, 3), num_classes=8, lr=0.01, alpha=1.0):
    image = Input(shape=input_shape)
    X = Conv2D(filters=16, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', name='conv_init')(image)
    X = bregnet_block(X, num_modules=13, num_filters=[16, 32], block_name='a', alpha=alpha)
    X = downsampling(X, num_filters=32, block_name='a', alpha=alpha)
    X = bregnet_block(X, num_modules=12, num_filters=[32, 64], block_name='b', alpha=alpha)
    X = downsampling(X, num_filters=64, block_name='b', alpha=alpha)
    X = bregnet_block(X, num_modules=12, num_filters=[64, 128], block_name='c', alpha=alpha)
    X = downsampling(X, num_filters=128, block_name='c', alpha=alpha)
    X = GlobalAveragePooling2D()(X)
    pred = Dense(num_classes, activation='softmax')(X)
    model = Model(inputs=image, outputs=pred)
    model.compile(Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def BREGNetCov(input_shape=(192, 192, 3), num_classes=8, lr=0.01, alpha=1.0):
    image = Input(shape=input_shape)
    X = Conv2D(filters=16, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal', name='conv_init')(image)
    X = bregnet_block(X, num_modules=13, num_filters=[16, 32], block_name='a', alpha=alpha)
    X = downsampling(X, num_filters=32, block_name='a', alpha=alpha)
    X = bregnet_block(X, num_modules=12, num_filters=[32, 64], block_name='b', alpha=alpha)
    X = downsampling(X, num_filters=64, block_name='b', alpha=alpha)
    X = bregnet_block(X, num_modules=12, num_filters=[64, 128], block_name='c', alpha=alpha)
    X = downsampling(X, num_filters=128, block_name='c', alpha=alpha)
    X = GlobalCovPooling2D(num_iter=5)(X)
    X = Flatten()(X)
    pred = Dense(num_classes, activation='softmax')(X)
    model = Model(inputs=image, outputs=pred)
    model.compile(Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
