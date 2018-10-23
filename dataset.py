import tensorflow as tf
from engine import get_from_config
import pandas as pd
import numpy as np

batch_size = get_from_config('batch_size')
num_of_threads = get_from_config('num_of_threads')

NUM_OF_CLASSES = 10

def load_mnist(is_training=True):
  """Loads the MNIST dataset

  :param is_training: A bool, used as a flag to indicate if we are
    in the training phase or no
  :return: Returns:
    - train data, train labels, number of train batches, val data,   if is_training
      val labels, number of val batches
    - test data, number of test batches                              otherwise
  """
  if is_training:
    train_data_path = get_from_config('train_data_path')
    assert train_data_path is not None

    train_data = pd.read_csv(train_data_path)

    # total number of training data
    num_of_data = train_data.shape[0]

    # validation dataset size
    val_sz = int(num_of_data * 0.2)

    train_X = train_data.iloc[val_sz:, 1:].values.astype('float32')
    train_Y = train_data.iloc[val_sz:, 0].values.astype('int32')

    # validation dataset
    val_X = train_data.iloc[:val_sz, 1:].values.astype('float32')
    val_Y = train_data.iloc[:val_sz, 0].values.astype('int32')

    # normalize images
    train_X = train_X / 255.0
    val_X = val_X / 255.0

    # reshape images
    train_X = train_X.reshape(train_X.shape[0], 28, 28, 1)
    val_X = val_X.reshape(val_X.shape[0], 28, 28, 1)

    # convert labels to one hot encoded vectors
    train_Y = tf.one_hot(train_Y, depth=NUM_OF_CLASSES, axis=1)
    val_Y = tf.one_hot(val_Y, depth=NUM_OF_CLASSES, axis=1)

    train_batch_num = (num_of_data-val_sz) // batch_size
    val_batch_num = (val_sz) // batch_size

    return train_X, train_Y, train_batch_num, val_X, val_Y, val_batch_num
  else:
    test_data_path = get_from_config('test_data_path')
    assert test_data_path is not None

    test_data = pd.read_csv(test_data_path)

    test_X = test_data.values.astype('float32')

    test_X /= 255.0
    test_X = test_X.reshape(test_X.shape[0], 28, 28, 1)

    test_X = np.squeeze(test_X)

    test_batch_num = test_data.shape[0] // batch_size

    return test_X, test_batch_num