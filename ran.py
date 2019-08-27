from tensorflow.python.keras.layers import Input, Multiply, Activation, Flatten, Lambda, Dense, Add, Concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam
from holonet import get_holonet_feature_extractor
from lightcnn import low_level_edge_detector, dsrc_module


def extract_crops(tensor):
    mode = 'color' if tensor.shape[-1] == 18 else 'gray'
    channels_per_image = 3 if mode == 'color' else 1
    imgs = []
    for i in range(6):
        imgs.append(tensor[:, :, :, channels_per_image * i:channels_per_image * (i + 1)])
    return imgs


def lambda_divide(params):
    return params[0] / params[1]


def extract_features(params):
    CNN = params[0]
    crops = params[1]
    return [CNN(x) for x in crops]


def attention_extractor(features):
    weight = Dense(1)(features)
    weight = Activation('sigmoid')(weight)
    return weight


def self_attention_module(crops, CNN):
    attention_weights = []
    for i in range(len(crops)):
        crops[i] = CNN(crops[i])
        weight = attention_extractor(crops[i])
        crops[i] = Multiply()([crops[i], weight])
        attention_weights.append(weight)
    total_weights = Add()(attention_weights)
    numerator = Add()(crops)  # numerator of fraction in definition of F_m (see paper)
    global_feature_representation = Lambda(lambda_divide)([numerator, total_weights])
    return global_feature_representation


def relation_attention_module(glob_fts, fts):
    attention_weights = []
    for i in range(len(fts)):
        fts[i] = Concatenate()([fts[i], glob_fts])
        weight = attention_extractor(fts[i])
        fts[i] = Multiply()([fts[i], weight])
        attention_weights.append(weight)
    total_weights = Add()(attention_weights)
    numerator = Add()(fts)  # numerator of fraction in definition of P_ran (see paper)
    final_representation = Lambda(lambda_divide)([numerator, total_weights])
    return final_representation


def get_ran(CNN, regularizers=(0.2, 0.2), input_shape=(192, 192, 18)):
    """ My implementation of the Region Attention Network (RAN) that was proposed by Wang et al. in the following
     article: https://arxiv.org/abs/1905.04075"""
    image = Input(shape=input_shape)
    crops = Lambda(extract_crops)(image)
    fts = Lambda(extract_features)([CNN, crops])
    glob_fts = self_attention_module(crops, CNN)
    features = relation_attention_module(glob_fts, fts)
    features = Dense(1024, kernel_regularizer=l2(regularizers[0]))(features)
    features = Dense(8, kernel_regularizer=l2(regularizers[1]))(features)
    prediction = Activation('softmax')(features)
    return Model(inputs=image, outputs=prediction)


def get_ran_holonet():
    """ RAN using Holonet as feature extractor. """
    return get_ran(get_holonet_feature_extractor(), regularizers=(0.2, 0.2), input_shape=(192, 192, 18))


def RANLCNN():
    """ Combination of RAN and LightCNN. """
    image = Input(shape=(192, 192, 3))
    X = low_level_edge_detector(image)
    X = dsrc_module(X, num_filters=16, module_name='a')
    X = dsrc_module(X, num_filters=32, module_name='b')
    X = dsrc_module(X, num_filters=64, module_name='c')
    X = dsrc_module(X, num_filters=128, module_name='d')
    X = dsrc_module(X, num_filters=256, module_name='e')
    X = dsrc_module(X, num_filters=512, module_name='f')
    X = Flatten()(X)
    feature_exctractor = Model(inputs=image, outputs=X)
    return get_ran(feature_exctractor)
