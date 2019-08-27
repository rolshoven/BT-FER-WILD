from tensorflow.python.keras.layers import Input, Dropout, Conv2D
from tensorflow.python.keras.layers import BatchNormalization, Activation, MaxPooling2D
from tensorflow.python.keras.layers import Flatten, Lambda, Dense, Concatenate
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam
from layers import *
import numpy as np
import os

def get_initial_locnet_weights(output_size):
    """
        Initialize the transformation parameters to the identity transformation
    """
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((output_size, 6), dtype='float32')
    weights = [W, b.flatten()]
    return weights


def get_initial_attention_weights(output_size):
    """
        Initialize the transformation parameters to the identity transformation
    """
    b = np.zeros(3, dtype='float32')
    b[0] = 1
    W = np.zeros((output_size, 3), dtype='float32')
    weights = [W, b.flatten()]
    return weights


def generate_attention_matrix(params):
    # Transform [s, t_x, t_y] to [s, 0, t_x, 0, s, t_y]
    attention_transform = np.zeros((3, 6), dtype=np.float32)
    attention_transform[0, 0] = 1.
    attention_transform[0, 4] = 1.
    attention_transform[1, 2] = 1.
    attention_transform[2, 5] = 1.
    attention_transform = tf.convert_to_tensor(attention_transform)
    return tf.matmul(params, attention_transform)


def get_old_model():
    model = Sequential()
    # Block 1
    model.add(Conv2D(10, (3, 3), input_shape=[192, 192, 3]))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(10, (3, 3)))
    model.add(MaxPooling2D([2, 2]))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    # Block 2
    model.add(Conv2D(10, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Conv2D(10, (3, 3)))
    model.add(MaxPooling2D([2, 2]))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(50, kernel_initializer='glorot_uniform'))
    model.add(Dense(11, kernel_initializer='glorot_uniform'))
    model.add(Activation("softmax"))

    return model


def get_model():
    image = Input(shape=(192, 192, 3))

    # Localization network
    locnet = Conv2D(8, (3, 3), kernel_initializer='he_normal')(image)
    locnet = MaxPooling2D((2, 2))(locnet)
    locnet = BatchNormalization()(locnet)
    locnet = Activation('relu')(locnet)

    locnet = Conv2D(10, (3, 3), kernel_initializer='he_normal')(locnet)
    locnet = MaxPooling2D((2, 2))(locnet)
    locnet = BatchNormalization()(locnet)
    locnet = Activation('relu')(locnet)

    locnet = Flatten()(locnet)
    locnet = Dense(32)(locnet)
    locnet = Activation('relu')(locnet)
    weights = get_initial_locnet_weights(32)
    locnet = Dense(6, weights=weights, name='locnet_params')(locnet)

    # Feature extraction network
    fexnet = Conv2D(10, (3, 3), kernel_initializer='he_normal')(image)
    fexnet = BatchNormalization()(fexnet)
    fexnet = Activation("relu")(fexnet)
    fexnet = Conv2D(10, (3, 3), kernel_initializer='he_normal')(fexnet)
    fexnet = MaxPooling2D([2, 2])(fexnet)
    fexnet = BatchNormalization()(fexnet)
    fexnet = Activation("relu")(fexnet)

    fexnet = Conv2D(10, (3, 3), kernel_initializer='he_normal')(fexnet)
    fexnet = BatchNormalization()(fexnet)
    fexnet = Activation("relu")(fexnet)
    fexnet = Conv2D(10, (3, 3), kernel_initializer='he_normal')(fexnet)
    fexnet = MaxPooling2D([2, 2])(fexnet)
    fexnet = BatchNormalization()(fexnet)
    fexnet = Activation("relu")(fexnet)

    # Transformation using sampling grid
    warped = Transformer((30, 30), name='sampler')([fexnet, locnet])

    # Final fully-connected layers with softmax activation
    warped = Flatten()(warped)

    warped = Dense(50, kernel_regularizer=l2(0.05))(warped)
    warped = Dense(8, kernel_regularizer=l2(0.05))(warped)
    prediction = Activation("softmax")(warped)

    return Model(inputs=image, outputs=prediction)


def get_model_attention():
    image = Input(shape=(192, 192, 3))

    # Localization network
    locnet = Conv2D(8, (3, 3))(image)
    locnet = MaxPooling2D((2, 2))(locnet)
    locnet = BatchNormalization()(locnet)
    locnet = Activation('relu')(locnet)

    locnet = Conv2D(10, (3, 3))(locnet)
    locnet = MaxPooling2D((2, 2))(locnet)
    locnet = BatchNormalization()(locnet)
    locnet = Activation('relu')(locnet)

    locnet = Flatten()(locnet)
    locnet = Dense(32)(locnet)
    locnet = Activation('relu')(locnet)
    weights = get_initial_attention_weights(32)
    locnet = Dense(3, weights=weights)(locnet)
    locnet = Lambda(generate_attention_matrix, name='locnet_params')(locnet)

    # Feature extraction network
    fexnet = Conv2D(10, (3, 3))(image)
    fexnet = BatchNormalization()(fexnet)
    fexnet = Activation("relu")(fexnet)
    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = MaxPooling2D([2, 2])(fexnet)
    fexnet = BatchNormalization()(fexnet)
    fexnet = Activation("relu")(fexnet)

    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = BatchNormalization()(fexnet)
    fexnet = Activation("relu")(fexnet)
    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = MaxPooling2D([2, 2])(fexnet)
    fexnet = BatchNormalization()(fexnet)
    fexnet = Activation("relu")(fexnet)

    # Transformation using sampling grid
    warped = Transformer((96, 96), name='sampler')([fexnet, locnet])

    # Final fully-connected layers with softmax activation
    warped = Flatten()(warped)

    warped = Dense(50, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.001))(warped)
    warped = Dense(11, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.001))(warped)
    prediction = Activation("softmax")(warped)

    return Model(inputs=image, outputs=prediction)


def get_model_dropout():
    image = Input(shape=(192, 192, 3))

    # Localization network
    locnet = Conv2D(8, (3, 3))(image)
    locnet = MaxPooling2D((2, 2))(locnet)
    locnet = Activation('relu')(locnet)

    locnet = Conv2D(10, (3, 3))(locnet)
    locnet = MaxPooling2D((2, 2))(locnet)
    locnet = Activation('relu')(locnet)

    locnet = Flatten()(locnet)
    locnet = Dense(32)(locnet)
    locnet = Activation('relu')(locnet)
    weights = get_initial_locnet_weights(32)
    locnet = Dense(6, weights=weights, name='locnet_params')(locnet)

    # Feature extraction network
    fexnet = Conv2D(10, (3, 3))(image)
    fexnet = Activation("relu")(fexnet)
    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = MaxPooling2D([2, 2])(fexnet)
    fexnet = Activation("relu")(fexnet)

    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = Activation("relu")(fexnet)
    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = MaxPooling2D([2, 2])(fexnet)
    fexnet = Activation("relu")(fexnet)
    fexnet = Dropout(0.3)(fexnet)

    # Transformation using sampling grid
    warped = Transformer((30, 30), name='sampler')([fexnet, locnet])

    # Final fully-connected layers with softmax activation
    warped = Flatten()(warped)

    warped = Dense(50, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.001))(warped)
    warped = Dense(11, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.001))(warped)
    prediction = Activation("softmax")(warped)

    return Model(inputs=image, outputs=prediction)


def get_model_dropout_attention(freeze_stn=False, freeze_fexnet=False, input_shape=(192, 192, 3), sampling_shape=(96, 96)):
    stn_trainable = not freeze_stn
    fexnet_trainable = not freeze_fexnet
    image = Input(shape=input_shape)

    # Localization network
    locnet = Conv2D(8, (3, 3), kernel_initializer='he_normal', trainable=stn_trainable)(image)
    locnet = MaxPooling2D((2, 2), trainable=stn_trainable)(locnet)
    locnet = Activation('relu', trainable=stn_trainable)(locnet)

    locnet = Conv2D(10, (3, 3), kernel_initializer='he_normal', trainable=stn_trainable)(locnet)
    locnet = MaxPooling2D((2, 2), trainable=stn_trainable)(locnet)
    locnet = Activation('relu', trainable=stn_trainable)(locnet)

    locnet = Flatten(trainable=stn_trainable)(locnet)
    locnet = Dense(32, trainable=stn_trainable)(locnet)
    locnet = Activation('relu', trainable=stn_trainable)(locnet)
    weights = get_initial_attention_weights(32)
    locnet = Dense(3, weights=weights, trainable=stn_trainable)(locnet)
    locnet = Lambda(generate_attention_matrix, name='locnet_params', trainable=stn_trainable)(locnet)

    # Feature extraction network
    fexnet = Conv2D(10, (3, 3), kernel_initializer='he_normal', trainable=fexnet_trainable)(image)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)
    fexnet = Conv2D(10, (3, 3), kernel_initializer='he_normal', trainable=fexnet_trainable)(fexnet)
    fexnet = MaxPooling2D([2, 2], trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)

    fexnet = Conv2D(10, (3, 3), kernel_initializer='he_normal', trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)
    fexnet = Conv2D(10, (3, 3), kernel_initializer='he_normal', trainable=fexnet_trainable)(fexnet)
    fexnet = MaxPooling2D([2, 2], trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)

    fexnet = Dropout(rate=0.4, trainable=fexnet_trainable)(fexnet)

    # Transformation using sampling grid
    warped = Transformer(sampling_shape, name='sampler')([fexnet, locnet])

    # Final fully-connected layers with softmax activation
    warped = Flatten()(warped)

    warped = Dense(50, kernel_regularizer=l2(0.3))(warped)
    warped = Dense(11, kernel_regularizer=l2(0.3))(warped)
    prediction = Activation("softmax")(warped)

    return Model(inputs=image, outputs=prediction)


def get_model_bn_attention():
    """
        Uses Batch Normalization instead of Dropout
    """
    image = Input(shape=(192, 192, 3))

    # Localization network
    locnet = Conv2D(8, (3, 3))(image)
    locnet = MaxPooling2D((2, 2))(locnet)
    locnet = Activation('relu')(locnet)

    locnet = Conv2D(10, (3, 3))(locnet)
    locnet = MaxPooling2D((2, 2))(locnet)
    locnet = Activation('relu')(locnet)

    locnet = Flatten()(locnet)
    locnet = Dense(32)(locnet)
    locnet = Activation('relu')(locnet)
    weights = get_initial_attention_weights(32)
    locnet = Dense(3, weights=weights)(locnet)
    locnet = Lambda(generate_attention_matrix, name='locnet_params')(locnet)

    # Feature extraction network
    fexnet = Conv2D(10, (3, 3))(image)
    fexnet = Activation("relu")(fexnet)
    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = MaxPooling2D([2, 2])(fexnet)
    fexnet = Activation("relu")(fexnet)

    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = Activation("relu")(fexnet)
    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = MaxPooling2D([2, 2])(fexnet)
    fexnet = Activation("relu")(fexnet)
    fexnet = BatchNormalization()(fexnet)

    # Transformation using sampling grid
    warped = Transformer((96, 96), name='sampler')([fexnet, locnet])

    # Final fully-connected layers with softmax activation
    warped = Flatten()(warped)

    warped = Dense(50, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.001))(warped)
    warped = Dense(11, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.001))(warped)
    prediction = Activation("softmax")(warped)

    return Model(inputs=image, outputs=prediction)


def get_simple_model():
    image = Input(shape=(192, 192, 3))

    # Feature extraction network
    fexnet = Conv2D(10, (3, 3))(image)
    fexnet = Activation("relu")(fexnet)
    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = MaxPooling2D([2, 2])(fexnet)
    fexnet = Activation("relu")(fexnet)

    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = Activation("relu")(fexnet)
    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = MaxPooling2D([2, 2])(fexnet)
    fexnet = Activation("relu")(fexnet)
    fexnet = Dropout(0.3)(fexnet)

    # Final fully-connected layers with softmax activation
    flat = Flatten()(fexnet)

    flat = Dense(50, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.001))(flat)
    flat = Dense(11, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.001))(flat)
    prediction = Activation("softmax")(flat)

    return Model(inputs=image, outputs=prediction)


def get_model_attention_first(freeze_stn=False, freeze_fexnet=False):
    stn_trainable = not freeze_stn
    fexnet_trainable = not freeze_fexnet

    image = Input(shape=(192, 192, 3))

    # Localization network
    locnet = Conv2D(8, (3, 3), trainable=stn_trainable)(image)
    locnet = MaxPooling2D((2, 2), trainable=stn_trainable)(locnet)
    locnet = Activation('relu', trainable=stn_trainable)(locnet)

    locnet = Conv2D(10, (3, 3), trainable=stn_trainable)(locnet)
    locnet = MaxPooling2D((2, 2), trainable=stn_trainable)(locnet)
    locnet = Activation('relu', trainable=stn_trainable)(locnet)

    locnet = Flatten(trainable=stn_trainable)(locnet)
    locnet = Dense(32, trainable=stn_trainable)(locnet)
    locnet = Activation('relu', trainable=stn_trainable)(locnet)
    weights = get_initial_attention_weights(32)
    locnet = Dense(3, weights=weights, trainable=stn_trainable)(locnet)
    locnet = Lambda(generate_attention_matrix, name='locnet_params', trainable=stn_trainable)(locnet)

    # Transformation using sampling grid
    warped = Transformer((96, 96), name='sampler')([image, locnet])

    # Feature extraction network
    fexnet = Conv2D(10, (3, 3), trainable=fexnet_trainable)(warped)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)
    fexnet = Conv2D(10, (3, 3), trainable=fexnet_trainable)(fexnet)
    fexnet = MaxPooling2D([2, 2], trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)

    fexnet = Conv2D(10, (3, 3), trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)
    fexnet = Conv2D(10, (3, 3), trainable=fexnet_trainable)(fexnet)
    fexnet = MaxPooling2D([2, 2], trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)
    fexnet = Dropout(rate=0.4, trainable=fexnet_trainable)(fexnet)

    # Final fully-connected layers with softmax activation
    fexnet = Flatten()(fexnet)

    fexnet = Dense(50, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.03))(fexnet)
    fexnet = Dense(11, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.03))(fexnet)
    prediction = Activation("softmax")(fexnet)

    return Model(inputs=image, outputs=prediction)


def get_model_full_transformation_first():
    image = Input(shape=(192, 192, 3))

    # Localization network
    locnet = Conv2D(8, (3, 3))(image)
    locnet = MaxPooling2D((2, 2))(locnet)
    locnet = Activation('relu')(locnet)

    locnet = Conv2D(10, (3, 3))(locnet)
    locnet = MaxPooling2D((2, 2))(locnet)
    locnet = Activation('relu')(locnet)

    locnet = Flatten()(locnet)
    locnet = Dense(32)(locnet)
    locnet = Activation('relu')(locnet)
    weights = get_initial_locnet_weights(32)
    locnet = Dense(6, weights=weights, name='locnet_params')(locnet)

    # Transformation using sampling grid
    warped = Transformer((96, 96), name='sampler')([image, locnet])

    # Feature extraction network
    fexnet = Conv2D(10, (3, 3))(warped)
    fexnet = Activation("relu")(fexnet)
    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = MaxPooling2D([2, 2])(fexnet)
    fexnet = Activation("relu")(fexnet)

    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = Activation("relu")(fexnet)
    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = MaxPooling2D([2, 2])(fexnet)
    fexnet = Activation("relu")(fexnet)
    fexnet = Dropout(rate=0.4)(fexnet)

    # Final fully-connected layers with softmax activation
    fexnet = Flatten()(fexnet)

    fexnet = Dense(50, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.001))(fexnet)
    fexnet = Dense(11, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.001))(fexnet)
    prediction = Activation("softmax")(fexnet)

    return Model(inputs=image, outputs=prediction)


def get_model_dropout_attention_freeze(freeze_stn=True):
    trainable = not freeze_stn

    image = Input(shape=(192, 192, 3))

    # Localization network
    locnet = Conv2D(8, (3, 3), trainable=trainable)(image)
    locnet = MaxPooling2D((2, 2), trainable=trainable)(locnet)
    locnet = Activation('relu', trainable=trainable)(locnet)

    locnet = Conv2D(10, (3, 3), trainable=trainable)(locnet)
    locnet = MaxPooling2D((2, 2), trainable=trainable)(locnet)
    locnet = Activation('relu', trainable=trainable)(locnet)

    locnet = Flatten(trainable=trainable)(locnet)
    locnet = Dense(32, trainable=trainable)(locnet)
    locnet = Activation('relu', trainable=trainable)(locnet)
    weights = get_initial_attention_weights(32)
    locnet = Dense(3, weights=weights, trainable=trainable)(locnet)
    locnet = Lambda(generate_attention_matrix, name='locnet_params', trainable=trainable)(locnet)

    # Feature extraction network
    fexnet = Conv2D(10, (3, 3))(image)
    fexnet = Activation("relu")(fexnet)
    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = MaxPooling2D([2, 2])(fexnet)
    fexnet = Activation("relu")(fexnet)

    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = Activation("relu")(fexnet)
    fexnet = Conv2D(10, (3, 3))(fexnet)
    fexnet = MaxPooling2D([2, 2])(fexnet)
    fexnet = Activation("relu")(fexnet)
    fexnet = Dropout(rate=0.4)(fexnet)

    # Transformation using sampling grid
    warped = Transformer((96, 96), name='sampler')([fexnet, locnet])

    # Final fully-connected layers with softmax activation
    warped = Flatten()(warped)

    warped = Dense(50, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.001))(warped)
    warped = Dense(11, kernel_initializer='glorot_uniform', kernel_regularizer=l2(0.001))(warped)
    prediction = Activation("softmax")(warped)

    return Model(inputs=image, outputs=prediction)


def get_model_emotion_only(freeze_stn=False, freeze_fexnet=False, input_shape=(192, 192, 3), sampling_shape=(64, 64)):
    stn_trainable = not freeze_stn
    fexnet_trainable = not freeze_fexnet
    image = Input(shape=input_shape)

    # Localization network
    locnet = Conv2D(8, (3, 3), kernel_initializer='he_normal', trainable=stn_trainable)(image)
    locnet = MaxPooling2D((2, 2), trainable=stn_trainable)(locnet)
    locnet = Activation('relu', trainable=stn_trainable)(locnet)

    locnet = Conv2D(10, (3, 3), kernel_initializer='he_normal', trainable=stn_trainable)(locnet)
    locnet = MaxPooling2D((2, 2), trainable=stn_trainable)(locnet)
    locnet = Activation('relu', trainable=stn_trainable)(locnet)

    locnet = Flatten(trainable=stn_trainable)(locnet)
    locnet = Dense(32, trainable=stn_trainable)(locnet)
    locnet = Activation('relu', trainable=stn_trainable)(locnet)
    weights = get_initial_attention_weights(32)
    locnet = Dense(3, weights=weights, trainable=stn_trainable)(locnet)
    locnet = Lambda(generate_attention_matrix, name='locnet_params', trainable=stn_trainable)(locnet)

    # Feature extraction network
    fexnet = Conv2D(10, (3, 3), kernel_initializer='he_normal', trainable=fexnet_trainable)(image)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)
    fexnet = Conv2D(10, (3, 3), kernel_initializer='he_normal', trainable=fexnet_trainable)(fexnet)
    fexnet = MaxPooling2D([2, 2], trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)

    fexnet = Conv2D(10, (3, 3), kernel_initializer='he_normal', trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)
    fexnet = Conv2D(10, (3, 3), kernel_initializer='he_normal', trainable=fexnet_trainable)(fexnet)
    fexnet = MaxPooling2D([2, 2], trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)

    fexnet = Dropout(rate=0.45, trainable=fexnet_trainable)(fexnet)

    # Transformation using sampling grid
    warped = Transformer(sampling_shape, name='sampler')([fexnet, locnet])

    # Final fully-connected layers with softmax activation
    warped = Flatten()(warped)

    warped = Dense(50, kernel_regularizer=l2(0.6))(warped)
    warped = Dense(8, kernel_regularizer=l2(0.6))(warped)
    prediction = Activation("softmax")(warped)

    return Model(inputs=image, outputs=prediction)


def get_extended_model(freeze_stn=False, freeze_fexnet=False, input_shape=(192, 192, 3), sampling_shape=(64, 64)):
    stn_trainable = not freeze_stn
    fexnet_trainable = not freeze_fexnet
    image = Input(shape=input_shape)

    # Localization network
    locnet = Conv2D(8, (3, 3), kernel_initializer='he_normal', trainable=stn_trainable)(image)
    locnet = MaxPooling2D((2, 2), trainable=stn_trainable)(locnet)
    locnet = Activation('relu', trainable=stn_trainable)(locnet)

    locnet = Conv2D(10, (3, 3), kernel_initializer='he_normal', trainable=stn_trainable)(locnet)
    locnet = MaxPooling2D((2, 2), trainable=stn_trainable)(locnet)
    locnet = Activation('relu', trainable=stn_trainable)(locnet)

    locnet = Flatten(trainable=stn_trainable)(locnet)
    locnet = Dense(32, trainable=stn_trainable)(locnet)
    locnet = Activation('relu', trainable=stn_trainable)(locnet)
    weights = get_initial_attention_weights(32)
    locnet = Dense(3, weights=weights, trainable=stn_trainable)(locnet)
    locnet = Lambda(generate_attention_matrix, name='locnet_params', trainable=stn_trainable)(locnet)

    # Feature extraction network
    fexnet = Conv2D(10, (3, 3), kernel_initializer='he_normal', trainable=fexnet_trainable)(image)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)
    fexnet = Conv2D(10, (3, 3), kernel_initializer='he_normal', trainable=fexnet_trainable)(fexnet)
    fexnet = MaxPooling2D([2, 2], trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)

    fexnet = Conv2D(10, (3, 3), kernel_initializer='he_normal', trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)
    fexnet = Conv2D(10, (3, 3), kernel_initializer='he_normal', trainable=fexnet_trainable)(fexnet)
    fexnet = MaxPooling2D([2, 2], trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)

    fexnet = Conv2D(15, (3, 3), kernel_initializer='he_normal', trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)
    fexnet = Conv2D(15, (3, 3), kernel_initializer='he_normal', trainable=fexnet_trainable)(fexnet)
    fexnet = MaxPooling2D([2, 2], trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)

    fexnet = Dropout(rate=0.4, trainable=fexnet_trainable)(fexnet)

    # Transformation using sampling grid
    warped = Transformer(sampling_shape, name='sampler')([fexnet, locnet])

    # Final fully-connected layers with softmax activation
    warped = Flatten()(warped)

    warped = Dense(50, kernel_regularizer=l2(0.5))(warped)
    warped = Dense(8, kernel_regularizer=l2(0.5))(warped)
    prediction = Activation("softmax")(warped)

    return Model(inputs=image, outputs=prediction)

def get_model_emotion_only_same_padding(freeze_stn=False, freeze_fexnet=False, input_shape=(192, 192, 3), sampling_shape=(64, 64)):
    stn_trainable = not freeze_stn
    fexnet_trainable = not freeze_fexnet
    image = Input(shape=input_shape)

    # Localization network
    locnet = Conv2D(8, (3, 3), padding='same', kernel_initializer='he_normal', trainable=stn_trainable)(image)
    locnet = MaxPooling2D((2, 2), padding='same', trainable=stn_trainable)(locnet)
    locnet = Activation('relu', trainable=stn_trainable)(locnet)

    locnet = Conv2D(10, (3, 3), padding='same', kernel_initializer='he_normal', trainable=stn_trainable)(locnet)
    locnet = MaxPooling2D((2, 2), padding='same', trainable=stn_trainable)(locnet)
    locnet = Activation('relu', trainable=stn_trainable)(locnet)

    locnet = Flatten(trainable=stn_trainable)(locnet)
    locnet = Dense(32, trainable=stn_trainable)(locnet)
    locnet = Activation('relu', trainable=stn_trainable)(locnet)
    weights = get_initial_attention_weights(32)
    locnet = Dense(3, weights=weights, trainable=stn_trainable)(locnet)
    locnet = Lambda(generate_attention_matrix, name='locnet_params', trainable=stn_trainable)(locnet)

    # Feature extraction network
    fexnet = Conv2D(10, (3, 3), padding='same', kernel_initializer='he_normal', trainable=fexnet_trainable)(image)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)
    fexnet = Conv2D(10, (3, 3), padding='same', kernel_initializer='he_normal', trainable=fexnet_trainable)(fexnet)
    fexnet = MaxPooling2D([2, 2], padding='same', trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)

    fexnet = Conv2D(10, (3, 3), padding='same', kernel_initializer='he_normal', trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)
    fexnet = Conv2D(10, (3, 3), padding='same', kernel_initializer='he_normal', trainable=fexnet_trainable)(fexnet)
    fexnet = MaxPooling2D([2, 2], padding='same', trainable=fexnet_trainable)(fexnet)
    fexnet = Activation("relu", trainable=fexnet_trainable)(fexnet)

    fexnet = Dropout(rate=0.40, trainable=fexnet_trainable)(fexnet)

    # Transformation using sampling grid
    warped = Transformer(sampling_shape, name='sampler')([fexnet, locnet])

    # Final fully-connected layers with softmax activation
    warped = Flatten()(warped)

    warped = Dense(50, kernel_regularizer=l2(0.5))(warped)
    warped = Dense(8, kernel_regularizer=l2(0.5))(warped)
    prediction = Activation("softmax")(warped)

    return Model(inputs=image, outputs=prediction)

def stn_on_input(image):
    # Localization network
    locnet = Conv2D(8, (3, 3), padding='same', kernel_initializer='he_normal')(image)
    locnet = MaxPooling2D((2, 2), padding='same')(locnet)
    locnet = Activation('relu')(locnet)
    locnet = Conv2D(10, (3, 3), padding='same', kernel_initializer='he_normal')(locnet)
    locnet = MaxPooling2D((2, 2), padding='same')(locnet)
    locnet = Activation('relu')(locnet)
    locnet = Flatten()(locnet)
    locnet = Dense(32)(locnet)
    locnet = Activation('relu')(locnet)
    weights = get_initial_attention_weights(32)
    locnet = Dense(3, weights=weights)(locnet)
    locnet = Lambda(generate_attention_matrix, name='locnet_params')(locnet)

    # Transformation using sampling grid
    warped = Transformer((30, 30), name='sampler')([image, locnet])

    return warped

def SHCNN(input_shape=(192, 192, 3), num_classes=8, initial_lr=0.01, alpha=0.02):
    input = Input(shape=input_shape)
    X = Conv2D(44, (5, 5), padding='same', kernel_initializer='he_normal')(input)
    X = MaxPooling2D((2, 2), padding='same')(X)
    X = LeakyReLU(alpha=alpha)(X)
    X = Conv2D(44, (3, 3), padding='same', kernel_initializer='he_normal')(X)
    X = MaxPooling2D((2, 2), padding='same')(X)
    X = LeakyReLU(alpha=alpha)(X)
    X = Conv2D(88, (5, 5), padding='same', kernel_initializer='he_normal')(X)
    X = MaxPooling2D((2, 2), padding='same')(X)
    X = LeakyReLU(alpha=alpha)(X)
    X = Flatten()(X)
    X = Dropout(rate=0.40)(X)
    X = Dense(2048)(X)
    X = LeakyReLU(alpha=alpha)(X)
    X = Dropout(rate=0.40)(X)
    X = Dense(1024)(X)
    X = LeakyReLU(alpha=alpha)(X)
    X = Dense(num_classes)(X)
    output = Activation('softmax')(X)
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=Adam(lr=initial_lr),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    return model

def FourfoldFER(input_shape=(192, 192, 3), num_classes=8, initial_lr=0.01):
    input = Input(shape=input_shape)
    X = Conv2D(16, (7, 7), padding='same', kernel_initializer='he_normal')(input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(16, (6, 6), padding='same', kernel_initializer='he_normal')(X)
    X = MaxPooling2D((2, 2), padding='same')(X)
    X = BatchNormalization()(X)
    Z1 = Activation('relu')(X)

    X = Conv2D(32, (5, 5), padding='same', kernel_initializer='he_normal')(Z1)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(32, (4, 4), padding='same', kernel_initializer='he_normal')(X)
    X = MaxPooling2D((2, 2), padding='same')(X)
    X = BatchNormalization()(X)
    Z2 = Activation('relu')(X)

    X = Conv2D(64, (4, 4), padding='same', kernel_initializer='he_normal')(Z2)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(X)
    X = MaxPooling2D((2, 2), padding='same')(X)
    X = BatchNormalization()(X)
    Z3 = Activation('relu')(X)

    X = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(Z3)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(128, (2, 2), padding='same', kernel_initializer='he_normal')(X)
    X = MaxPooling2D((2, 2), padding='same')(X)
    X = BatchNormalization()(X)
    Z4 = Activation('relu')(X)

    Z1 = GlobalCovPooling2D(num_iter=5)(Z1)
    Z1 = Dense(8)(Z1)
    Z1 = Activation('relu')(Z1)

    Z2 = GlobalCovPooling2D(num_iter=5)(Z2)
    Z2 = Dense(8)(Z2)
    Z2 = Activation('relu')(Z2)

    Z3 = GlobalCovPooling2D(num_iter=5)(Z3)
    Z3 = Dense(8)(Z3)
    Z3 = Activation('relu')(Z3)

    Z4 = GlobalCovPooling2D(num_iter=5)(Z4)
    Z4 = Dense(8)(Z4)
    Z4 = Activation('relu')(Z4)

    Z = Concatenate()([Z1, Z2, Z3, Z4])
    Z = Dense(8)(Z)
    Z = Activation('softmax')(Z)

    model = Model(inputs=input, outputs=Z)
    model.compile(optimizer=Adam(lr=initial_lr),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    return model
