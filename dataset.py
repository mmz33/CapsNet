import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np


def load_mnist(val_prc=0.1):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # load mnist images

    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # normalize images to have pixels in range [0.0, 1.0]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    val_size = int(x_train.shape[0] * val_prc)
    assert val_size > 0, 'val_size should be > 0, received: {}'.format(val_size)

    x_val = x_train[:val_size]
    y_val = y_train[:val_size]

    x_train = x_train[val_size:]
    y_train = y_train[val_size:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def plot_img(img, label=None):
    """
    Plot MNIST image

    :param img: MNIST image of shape either (784,) or (28, 28)
    :param label: An integer, the corresponding label for `img`
    """
    assert img.ndim == 1 or img.ndim == 2, 'img should have shape (H,W) or (H*W,)'
    import numpy
    assert numpy.prod([img.shape[i] for i in range(img.ndim)]) == 784, 'img should have 784 pixels'
    if img.ndim == 1:
        img = img.reshape(28, 28)
    plt.imshow(img, cmap='gray')
    if label:
        plt.title('Label %i' % label)
    plt.show()
