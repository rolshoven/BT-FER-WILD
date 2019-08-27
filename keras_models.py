from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.applications import Xception as xcpt
from tensorflow.python.keras.applications import InceptionResNetV2 as irnv2
from tensorflow.python.keras.applications import DenseNet121 as dn121
from keras_vggface.vggface import VGGFace
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Activation
from tensorflow.python.keras.layers import Conv2D, BatchNormalization
from tensorflow.python.keras.layers import Dropout, Input
from tensorflow.python.keras.optimizers import Adam
from layers import GlobalCovPooling2D, Transformer
import numpy as np


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


def RN50(input_shape=(192, 192, 3), num_classes=8, lr=0.001):
    resnet50 = ResNet50(include_top=False, weights=None, input_shape=input_shape, pooling='avg')
    out = resnet50.output
    out = Dense(8, activation='softmax')(out)
    model = Model(inputs=resnet50.input, outputs=out)
    model.compile(optimizer=Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def RN50CovPool(input_shape=(192, 192, 3), num_classes=8, lr=0.001):
    resnet50 = ResNet50(include_top=False, weights=None, input_shape=input_shape)
    out = resnet50.output
    out = Conv2D(256, (1, 1))(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = GlobalCovPooling2D(num_iter=5)(out)
    out = Flatten()(out)
    out = Dense(8)(out)
    out = Activation('softmax')(out)
    model = Model(inputs=resnet50.input, outputs=out)
    model.compile(optimizer=Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def VGGFTrans(input_shape=(197, 197, 3), num_classes=8, lr=0.001):
    vggface = VGGFace(model='resnet50', include_top=False, input_shape=input_shape, pooling='avg')
    last_layer = vggface.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(num_classes, activation='softmax', name='classifier')(x)

    # Freeze the weights of the first layers
    i = 0
    layer = vggface.layers[i]
    while layer.name != 'conv5_1_1x1_reduce':
        layer.trainable = False
        i += 1
        layer = vggface.layers[i]

    model = Model(vggface.input, out)
    model.compile(optimizer=Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def Xception(input_shape=(192, 192, 3), num_classes=8, lr=0.001):
    xception = xcpt(include_top=True, classes=8, weights=None, input_shape=input_shape, pooling='avg')
    xception.compile(optimizer=Adam(lr=lr),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
    return xception


def XceptionCov(input_shape=(192, 192, 3), num_classes=8, num_iter=5):
    xcptcov = xcpt(include_top=False, weights=None, input_shape=input_shape)
    out = xcptcov.layers[-7].output
    out = GlobalCovPooling2D(num_iter=num_iter)(out)
    out = Dense(num_classes)(out)
    out = Activation('softmax')(out)
    model = Model(inputs=xcptcov.input, outputs=out)
    return model


def InceptionResNetV2(input_shape=(192, 192, 3), num_classes=8, lr=0.001):
    inres = irnv2(include_top=True, classes=8, weights=None, input_shape=input_shape, pooling='avg')
    inres.compile(optimizer=Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return inres


def DenseNet121(input_shape=(192, 192, 3), num_classes=8, lr=0.001):
    dnet = dn121(include_top=True, classes=8, weights=None, input_shape=input_shape, pooling='avg')
    dnet.compile(optimizer=Adam(lr=lr),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    return dnet


def DenseNet121Cov(input_shape=(192, 192, 3), num_classes=8, num_iter=5):
    dnetcov = dn121(include_top=False, weights=None, input_shape=input_shape)
    out = dnetcov.output
    out = Conv2D(512, (1, 1), kernel_initializer='he_normal')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = GlobalCovPooling2D(num_iter=num_iter)(out)
    out = Dense(num_classes)(out)
    out = Activation('softmax')(out)
    model = Model(inputs=dnetcov.input, outputs=out)
    return model


def DenseNet121CovDropout(input_shape=(192, 192, 3), num_classes=8, num_iter=5):
    dnetcov = dn121(include_top=False, weights=None, input_shape=input_shape)
    out = dnetcov.output
    out = Conv2D(512, (1, 1), kernel_initializer='he_normal')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = GlobalCovPooling2D(num_iter=num_iter)(out)
    out = Dropout(rate=0.5)(out)
    out = Dense(num_classes)(out)
    out = Activation('softmax')(out)
    model = Model(inputs=dnetcov.input, outputs=out)
    return model


def STNDenseNet121CovDropout(input_shape=(192, 192, 3), num_classes=8, num_iter=5):
    image = Input(shape=input_shape)

    # Localization network
    locnet = Conv2D(8, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(image)
    locnet = Activation('relu')(locnet)

    locnet = Conv2D(10, (4, 4), strides=(2, 2), padding='same', kernel_initializer='he_normal')(locnet)
    locnet = Activation('relu')(locnet)

    locnet = Flatten()(locnet)
    locnet = Dense(32)(locnet)
    locnet = Activation('relu')(locnet)
    locnet = Dense(6, weights=get_initial_locnet_weights(32), name='locnet_params')(locnet)

    # Transformation using sampling grid
    warped = Transformer((input_shape[0], input_shape[1]), name='sampler')([image, locnet])
    dnetcov = dn121(include_top=False, weights=None, input_tensor=warped)

    out = dnetcov.output
    out = Conv2D(512, (1, 1), kernel_initializer='he_normal')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = GlobalCovPooling2D(num_iter=num_iter)(out)
    out = Dropout(rate=0.5)(out)
    out = Dense(num_classes)(out)
    out = Activation('softmax')(out)
    model = Model(inputs=dnetcov.input, outputs=out)
    return model
