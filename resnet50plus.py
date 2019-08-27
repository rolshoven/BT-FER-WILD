from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.layer_utils import get_source_inputs
from layers import GlobalCovPooling2D


def identity_block(input_tensor, kernel_size, filters, stage, block, alpha):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    # x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.ELU(alpha=alpha)(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    # x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.ELU(alpha=alpha)(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               alpha,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    # x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.ELU(alpha=alpha)(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    # x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.ELU(alpha=alpha)(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    return x


def ResNet50Plus(include_top=False,
                 input_tensor=None,
                 input_shape=(192, 192, 3),
                 alpha=1.0,
                 pooling='cov',
                 classes=8,
                 **kwargs):
    """ Adaption of ResNet50 that uses ELU instead of ReLU and uses less batch normalization layers. """

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.ELU(alpha=alpha)(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', alpha=alpha, strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', alpha=alpha)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', alpha=alpha)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', alpha=alpha)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', alpha=alpha)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', alpha=alpha)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', alpha=alpha)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', alpha=alpha)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', alpha=alpha)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', alpha=alpha)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', alpha=alpha)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', alpha=alpha)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', alpha=alpha)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', alpha=alpha)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', alpha=alpha)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', alpha=alpha)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        elif pooling == 'cov':
            # Reduce number of channels before applying covariance pooling
            # num_channels = int(np.ceil(np.max(np.roots([1, 1, -2*classes]))))
            x = layers.Conv2D(classes, (1, 1),
                              padding='valid',
                              kernel_initializer='he_normal',
                              name='reduce_channels')(x)
            x = GlobalCovPooling2D(num_iter=5)(x)
            x = layers.Dense(classes, activation='softmax', name='fc{}'.format(classes))(x)
        else:
            warnings.warn('The output shape of `ResNet50(include_top=False)` '
                          'has been changed since Keras 2.2.0.')

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet50plus')

    return model
