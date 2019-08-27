from tensorflow.python.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Lambda, \
    Dense, Add, Concatenate, UpSampling2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam
from tensorflow.nn import crelu
from layers import GlobalCovPooling2D
from losses import island_crossentropy_loss
from model import stn_on_input


def phase_convolutional_block(X):
    X = Conv2D(8, (7, 7), padding='same', kernel_initializer='he_normal')(X)
    X = BatchNormalization()(X)
    X = Lambda(crelu)(X)
    X = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)
    return X


def phase_residual_block_1(X):
    X_main = Conv2D(12, (1, 1), padding='same', kernel_initializer='he_normal')(X)
    X_main = BatchNormalization()(X_main)
    X_main = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(X_main)
    X_main = BatchNormalization()(X_main)
    X_main = Lambda(crelu)(X_main)
    X_main = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(X_main)
    X_main = BatchNormalization()(X_main)
    X_res_branch = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(X)
    X_res_branch = BatchNormalization()(X_res_branch)
    X_combined = Add()([X_main, X_res_branch])
    X_main = Conv2D(12, (1, 1), padding='same', kernel_initializer='he_normal')(X_combined)
    X_main = BatchNormalization()(X_main)
    X_main = Conv2D(12, (3, 3), padding='same', kernel_initializer='he_normal')(X_main)
    X_main = BatchNormalization()(X_main)
    X_main = Lambda(crelu)(X_main)
    X_main = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(X_main)
    X_main = BatchNormalization()(X_main)
    X_combined = Add()([X_main, X_combined])
    return X_combined


def phase_residual_block_2(X):
    X_main = Conv2D(16, (1, 1), strides=(2, 2), padding='same', kernel_initializer='he_normal')(X)
    X_main = BatchNormalization()(X_main)
    X_main = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(X_main)
    X_main = BatchNormalization()(X_main)
    X_main = Lambda(crelu)(X_main)
    X_main = Conv2D(48, (1, 1), padding='same', kernel_initializer='he_normal')(X_main)
    X_main = BatchNormalization()(X_main)
    X_res_branch = Conv2D(48, (1, 1), strides=(2, 2), padding='same', kernel_initializer='he_normal')(X)
    X_res_branch = BatchNormalization()(X_res_branch)
    X_combined = Add()([X_main, X_res_branch])
    X_main = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(X_combined)
    X_main = BatchNormalization()(X_main)
    X_main = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(X_main)
    X_main = BatchNormalization()(X_main)
    X_main = Lambda(crelu)(X_main)
    X_main = Conv2D(48, (1, 1), padding='same', kernel_initializer='he_normal')(X_main)
    X_main = BatchNormalization()(X_main)
    X_combined = Add()([X_main, X_combined])
    return X_combined


def inception_residual_block(X):
    X1 = Conv2D(24, (1, 1), strides=(2, 2), padding='same', kernel_initializer='he_normal')(X)
    X1 = BatchNormalization()(X1)
    X2 = Conv2D(16, (1, 1), strides=(2, 2), padding='same', kernel_initializer='he_normal')(X)
    X2 = BatchNormalization()(X2)
    X2 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(X2)
    X2 = BatchNormalization()(X2)
    X3 = Conv2D(12, (1, 1), strides=(2, 2), padding='same', kernel_initializer='he_normal')(X)
    X3 = BatchNormalization()(X3)
    X3 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(X3)
    X3 = BatchNormalization()(X3)
    X3 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(X3)
    X3 = BatchNormalization()(X3)
    X4 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)
    X4 = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(X4)
    X4 = BatchNormalization()(X4)
    concat = Concatenate()([X1, X2, X3, X4])
    concat = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(concat)
    concat = BatchNormalization()(concat)
    X_res_branch = Conv2D(64, (1, 1), strides=(2, 2), padding='same', kernel_initializer='he_normal')(X)
    X_res_branch = BatchNormalization()(X_res_branch)
    X_combined = Add()([concat, X_res_branch])
    X_combined = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(X_combined)
    return X_combined


def HoloNet(input_shape=(192, 192, 3), regularizers=(0.1, 0.1), classes=8):
    image = Input(shape=input_shape)
    X = phase_convolutional_block(image)
    X = phase_residual_block_1(X)
    X = phase_residual_block_2(X)
    X = inception_residual_block(X)
    X = Flatten()(X)
    X = Dense(1024, kernel_regularizer=l2(regularizers[0]))(X)
    X = Dense(classes, kernel_regularizer=l2(regularizers[1]))(X)
    prediction = Activation('softmax')(X)
    return Model(inputs=image, outputs=prediction)


def HoloNetCov(input_shape=(192, 192, 3), regularizers=(0.1, 0.1), classes=8):
    """ Holonet combined with covariance pooling. """
    image = Input(shape=input_shape)
    X = phase_convolutional_block(image)
    X = phase_residual_block_1(X)
    X = phase_residual_block_2(X)
    X = inception_residual_block(X)
    X = GlobalCovPooling2D(num_iter=5)(X)
    X = Dense(classes, kernel_regularizer=l2(regularizers[1]))(X)
    prediction = Activation('softmax')(X)
    return Model(inputs=image, outputs=prediction)


def get_holonet_stn(input_shape=(192, 192, 3), regularizers=(0.2, 0.2)):
    """ A Holonet with a spatial transformer. """
    image = Input(shape=input_shape)
    X = stn_on_input(image)
    X = phase_convolutional_block(X)
    X = phase_residual_block_1(X)
    X = phase_residual_block_2(X)
    X = inception_residual_block(X)
    X = Flatten()(X)
    X = Dense(1024, kernel_regularizer=l2(regularizers[0]))(X)
    X = Dense(8, kernel_regularizer=l2(regularizers[1]))(X)
    prediction = Activation('softmax')(X)
    return Model(inputs=image, outputs=prediction)


def get_holonet_feature_extractor(input_shape=(192, 192, 3)):
    """ Holonet feature extractor without classification module. """
    image = Input(shape=input_shape)
    X = phase_convolutional_block(image)
    X = phase_residual_block_1(X)
    X = phase_residual_block_2(X)
    X = inception_residual_block(X)
    X = Flatten()(X)
    return Model(inputs=image, outputs=X)


def HoloNetIL(input_shape=(192, 192, 3), num_classes=8, regularizers=(0.2, 0.2), lr=0.01, balance=0.01):
    """ Holonet with Island Loss. """
    image = Input(shape=input_shape)
    X = stn_on_input(image)
    X = phase_convolutional_block(X)
    X = phase_residual_block_1(X)
    X = phase_residual_block_2(X)
    X = inception_residual_block(X)
    X = Flatten()(X)
    fts = Dense(1024, kernel_regularizer=l2(regularizers[0]))(X)
    X = Dense(num_classes, kernel_regularizer=l2(regularizers[1]))(fts)
    prediction = Activation('softmax')(X)
    HoloNet = Model(inputs=image, outputs=prediction)
    HoloNet.compile(optimizer=Adam(lr=lr),
                    loss=island_crossentropy_loss(fts, num_classes=num_classes, balance=balance),
                    metrics=['accuracy'])
    return HoloNet
