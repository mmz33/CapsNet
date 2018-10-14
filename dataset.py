import tensorflow as tf
from engine import get_from_config
import numpy as np

batch_size = get_from_config('batch_size')
num_of_threads = get_from_config('num_of_threads')

# TODO: Read training and test data from files
# TODO: Split train data to validation data
# TODO: return the data and the number of batches
def load_mnist(is_training=True):
  if is_training:
    train_data_path = get_from_config('train_data_path')
    assert train_data_path is not None

    data = np.fromfile(train_data_path)
    pass

def get_batch_data():
  train_X, train_Y, train_batches, test_X, test_Y, test_batches = load_mnist()
  data_queues = tf.train.slice_input_producer([train_X, train_Y],
                                              name='slice_train_data')
  # create batches
  X, Y = tf.train.shuffle_batch(data_queues,
                                batch_size=batch_size,
                                capacity=batch_size * 64,
                                min_after_dequeue=batch_size * 32,
                                num_threads=num_of_threads,
                                name='shuffle_batch')
  return X, Y