import tensorflow as tf
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.models import Model
from layers import Transformer
from datetime import datetime

# Class labels
EMOTIONS = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger',
    7: 'Contempt'
}


def _get_confusion_matrix(ground_truth, predictions):
    """
        params:
            - ground_truth: array of shape (m, num_labels) containing the labels
                            for the m input images in one_hot encoding.
            - predictions:  array of shape (m, num_labels) containing the softmax
                            probability output for each of the m input images
        returns:
            - C: confusion matrix
    """
    y_true = np.array([np.where(r == 1)[0][0] for r in ground_truth])
    y_pred = np.argmax(predictions, axis=1)
    return confusion_matrix(y_true, y_pred)


def _apply_learned_transformation(image, stn):
    image = np.array([image])
    image = image.astype(np.float32)
    theta = stn.predict(image)
    theta = np.reshape(theta, (1, 2, 3))
    out = Transformer((192, 192))([image, theta])
    return tf.keras.backend.eval(out)[0]


def _get_stn(model):
    return Model(inputs=model.input,
                 outputs=model.get_layer('locnet_params').output)


def _transform(images, model, weights=None):
    if not weights is None:
        model.load_weights(weights)
    stn = _get_stn(model)
    originals = []
    transformed = []
    for img in images:
        originals.append(img)
        transformed.append(_apply_learned_transformation(img, stn))
    return originals, transformed


def plot_confusion_matrix(ground_truth, predictions, num_labels=8):
    """
        params:
            - ground_truth: array of shape (m, num_labels) containing the labels
                            for the m input images in one_hot encoding.
            - predictions:  array of shape (m, num_labels) containing the softmax
                            probability output for each of the m input images
    """
    cm = _get_confusion_matrix(ground_truth, predictions)
    df_cm = pd.DataFrame(cm, index=[EMOTIONS[i] for i in range(num_labels)],
                         columns=[EMOTIONS[i] for i in range(num_labels)])
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='g', cmap=sns.cm.rocket_r)
    plt.savefig('visualizations/confusion_matrix_{}.jpg'.format(datetime.now().timestamp()))


def plot_prediction(model, image_path, weights=None):
    if not weights is None:
        model.load_weights(weights)
    image = resize(imread(image_path), (192, 192))
    prediction = EMOTIONS[np.argmax(model.predict(np.array([image])))]
    plt.imshow(image)
    plt.title(prediction)
    plt.savefig('visualizations/prediction_{}.jpg'.format(datetime.now().timestamp()))


def plot_with_labels(images, labels, plot_dims):
    assert images.shape[0] == labels.shape[0]
    assert images.shape[0] <= plot_dims[0] * plot_dims[1]
    fig, ax = plt.subplots(nrows=plot_dims[0], ncols=plot_dims[1], figsize=(15, 15),
                           subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(left=0.03, right=0.97, hspace=0.1, wspace=0.05)
    for row in range(plot_dims[0]):
        for col in range(plot_dims[1]):
            idx = row * plot_dims[1] + col
            if idx < images.shape[0]:
                label = np.where(labels[idx] == 1.)[0][0]
                ax[row, col].imshow(images[idx])
                ax[row, col].set_title(EMOTIONS[label], fontsize=26)
    plt.tight_layout()
    plt.savefig('visualizations/images_with_labels_{}.jpg'.format(datetime.now().timestamp()))


def plot_transformations(images, model, weights=None):
    originals, transformed = _transform(images, model, weights=weights)
    fig, ax = plt.subplots(nrows=2, ncols=len(originals), figsize=(15, 15), subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(left=0.03, right=0.97, hspace=0.1, wspace=0.05)
    for col in range(len(originals)):
        ax[0, col].imshow(originals[col])
        ax[0, col].set_title('Original', fontsize=26)
    for col in range(len(originals)):
        ax[1, col].imshow(transformed[col])
        ax[1, col].set_title('Transformed', fontsize=26)
    plt.tight_layout()
    plt.savefig('visualizations/transformations_{}.jpg'.format(datetime.now().timestamp()))


def plot_generator_batch(generator, idx):
    imgs, _ = generator.__getitem__(idx)
    fig = plt.figure(figsize=(15, 15))
    size = int(np.ceil(np.sqrt(imgs.shape[0])))
    for i in range(int(imgs.shape[0])):
        ax = fig.add_subplot(size, size, i + 1)
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        plt.imshow(imgs[i])
    plt.savefig('visualizations/generator/batch_{}_{}.jpg'.format(idx, datetime.now().timestamp()))
