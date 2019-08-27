from tensorflow.python import keras
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import glorot_uniform
from tensorflow.python.keras.regularizers import l2
from layers import GlobalCovPooling2D
from losses import island_crossentropy_loss


def low_level_edge_detector(X, padding='same'):  # updated from valid to same
    """ Extracts low-level edge features of the image and retains details. """
    X = Conv2D(filters=8, kernel_size=(3, 3), padding=padding, name='conv_1')(X)
    X = BatchNormalization(name='bn_1')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), padding=padding, name='conv_2')(X)
    X = BatchNormalization(name='bn_2')(X)
    X = Activation('relu')(X)
    return X


def dsrc_module(X, num_filters, module_name, padding='same'):
    """Depthwise separable residual convolution module.

    Keyword arguments:
    X -- data of last layer
    num_filters -- number of filters to use in the DSRC module.
    module_name -- letter indicating the position of the module inside the network.
    """
    sconv_name_base = 'sconv_{}_'.format(module_name)
    conv_name_base = 'conv_{}'.format(module_name)
    bn_name_base = 'bn_{}_'.format(module_name)

    first_branch = SeparableConv2D(filters=num_filters, kernel_size=(1, 1), padding=padding,
                                   name=sconv_name_base + str(1))(X)
    first_branch = BatchNormalization(name=bn_name_base + str(1.1))(first_branch)
    first_branch = Activation('relu')(first_branch)
    first_branch = SeparableConv2D(filters=num_filters, kernel_size=(3, 3), padding=padding,
                                   name=sconv_name_base + str(2))(first_branch)
    first_branch = BatchNormalization(name=bn_name_base + str(1.2))(first_branch)
    first_branch = Activation('relu')(first_branch)
    first_branch = SeparableConv2D(filters=num_filters, kernel_size=(1, 1), padding=padding,
                                   name=sconv_name_base + str(3))(first_branch)
    first_branch = BatchNormalization(name=bn_name_base + str(1.3))(first_branch)
    first_branch = Activation('relu')(first_branch)
    first_branch = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding=padding)(first_branch)

    second_branch = Conv2D(filters=num_filters, kernel_size=(1, 1), padding=padding, strides=(2, 2),
                           name=conv_name_base)(X)
    second_branch = BatchNormalization(name=bn_name_base + str(2.1))(second_branch)
    second_branch = Activation('relu')(second_branch)
    combined = Add()([first_branch, second_branch])
    return combined


def classification_module(X, num_classes, padding='same', regularizer=None):
    X = Conv2D(filters=num_classes, kernel_size=(1, 1), padding=padding)(X)
    X = GlobalAveragePooling2D()(X)
    X = Flatten()(X)
    if regularizer is None:
        X = Dense(num_classes, name='fc' + str(num_classes))(X)
    else:
        X = Dense(num_classes, name='fc' + str(num_classes), kernel_regularizer=l2(regularizer))(X)
    X = Activation('softmax')(X)
    return X


def classification_module_cov(X, num_classes, regularizer=None):
    # X = Conv2D(filters=num_classes, kernel_size=(1,1), padding='same')(X)
    X = GlobalCovPooling2D()(X)
    X = Flatten()(X)
    if regularizer is None:
        X = Dense(num_classes, name='fc' + str(num_classes))(X)
    else:
        X = Dense(num_classes, name='fc' + str(num_classes), kernel_regularizer=l2(regularizer))(X)
    X = Activation('softmax')(X)
    return X


def LightCNN(input_shape=(192, 192, 3), num_classes=8, regularizer=None, lr=0.01):
    """ My implementation of the LightCNN proposed in this article:
     https://www.sciencedirect.com/science/article/pii/S0925231219306137
    """
    image = Input(shape=input_shape)
    X = low_level_edge_detector(image)
    X = dsrc_module(X, num_filters=16, module_name='a')
    X = dsrc_module(X, num_filters=32, module_name='b')
    X = dsrc_module(X, num_filters=64, module_name='c')
    X = dsrc_module(X, num_filters=128, module_name='d')
    X = dsrc_module(X, num_filters=256, module_name='e')
    X = dsrc_module(X, num_filters=512, module_name='f')
    predicted_class = classification_module(X, num_classes=num_classes, regularizer=regularizer)
    model = Model(inputs=image, outputs=predicted_class)
    model.compile(optimizer=keras.optimizers.Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def LightCNNCov(input_shape=(192, 192, 3), num_classes=8, regularizer=None, lr=0.01):
    """ LightCNN with covariance pooling. """
    image = Input(shape=input_shape)
    X = low_level_edge_detector(image)
    X = dsrc_module(X, num_filters=16, module_name='a')
    X = dsrc_module(X, num_filters=32, module_name='b')
    X = dsrc_module(X, num_filters=64, module_name='c')
    X = dsrc_module(X, num_filters=128, module_name='d')
    X = dsrc_module(X, num_filters=256, module_name='e')
    X = dsrc_module(X, num_filters=512, module_name='f')
    predicted_class = classification_module_cov(X, num_classes=num_classes, regularizer=regularizer)
    model = Model(inputs=image, outputs=predicted_class)
    model.compile(optimizer=keras.optimizers.Adam(lr=lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    print('--- LightCNN summary ---')
    LightCNN().summary()
    print('--- LightCNNCov summary ---')
    LightCNNCov().summary()
