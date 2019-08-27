from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Lambda, Concatenate

XCEPTION = load_model('/data/cvg/luca/FER/models/Xception/keras_models/val_acc_56.hdf5')
RESNET50 = load_model('/data/cvg/luca/FER/models/ResNet50/keras_models/val_acc_54.hdf5')
for layer in XCEPTION.layers:
    layer.trainable = False
for layer in RESNET50.layers:
    layer.trainable = False


def xception_features(image):
    return XCEPTION(image)


def resnet_features(image):
    return RESNET50(image)


def XceptionResNet50(input_shape=(192, 192, 3), classes=8):
    image = Input(shape=input_shape)
    xc_fts = Lambda(xception_features)(image)
    rn_fts = Lambda(resnet_features)(image)
    combined = Concatenate()([xc_fts, rn_fts])
    out = Dense(classes, activation='softmax')(combined)
    model = Model(inputs=image, outputs=out)
    return model
