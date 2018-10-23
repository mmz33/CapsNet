import tensorflow as tf
import dataset
import logging
import numpy as np

# Contains a set of param
config = {
  # For training
  'epochs': 10,
  'batch_size': 128,
  'epoch': 50,
  'routing_iterations': 3,
  'stddev': 0.01,
  'checkpoint_path': './my_capsnet',

  # For dataset
  'train_data_path': 'data/train.csv',
  'test_data_path' : 'data/test.csv',
  'num_of_threads': 4,

  # For margin loss
  'm_plus': 0.9,
  'm_minus': 0.1,
  'lambda': 0.5
}

def get_from_config(key):
  """Returns the value of the given key from the config dict

  :param key: A string, the name of the param in the config dict
  :return: The value of the key in config
  """

  assert isinstance(key, str), str(key) + ' must be of type str.'
  assert key in config, str(key) + ' is not found in config.'
  return config[key]

# global variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()
checkpoint_path = get_from_config('checkpoint_path')
batch_size = get_from_config('batch_size')

def train(model, restore_checkpoint=True):
  # loads the data
  train_X, train_Y, train_batch_num, val_X, val_Y, val_batch_num = dataset.load_mnist()

  num_of_epochs = get_from_config('epochs')

  # store the best loss so far which can be used in validation
  best_loss_val = np.inf
  with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
      saver.restore(sess, checkpoint_path)
    else:
      init.run()

    logging.info('Start Training...')
    for epoch in range(num_of_epochs):
      for train_iter in range(train_batch_num):

        start = train_iter * batch_size
        end = start + batch_size

        # Run the training operation and measure the loss
        _, global_step, train_loss, train_acc, _summary = \
          sess.run([model.train_op, model.global_step, model.total_loss,
                    model.accuracy, model.train_summary],
                    feed_dict={model.X: train_X[start:end],
                               model.Y: train_Y[start:end]})

        logging.info('Epoch {}, Iteration {}/{}, loss: {}, accuracy: {}'
          .format(epoch+1, train_iter+1, train_batch_num, train_loss, train_acc))

      # Do validation at the end of each epoch
      val_acc = []
      val_loss = []
      for val_iter in range(val_batch_num):
        logging.info('Validating...')

        start = val_iter * batch_size
        end = start + batch_size

        _, global_step, train_loss, train_acc, _summary = \
          sess.run([model.train_op, model.global_step, model.total_loss,
                    model.accuracy, model.train_summary],
                    feed_dict={model.X: val_X[start:end],
                               model.Y: val_Y[start:end]})

        logging.info('Validation: {}/{}'.format(val_iter, val_batch_num))

        val_acc.append(train_acc)
        val_loss.append(train_loss)

      total_acc = np.mean(val_acc)
      total_loss = np.mean(val_loss)

      logging.info('Validation Epoch {}, Val accuracy: {}, Loss: {}'
        .format(epoch+1, total_acc, total_loss))

      if best_loss_val > total_loss:
        saver.save(sess, checkpoint_path)
        best_loss_val = total_loss
      else:
        logging.info('Error not improving. Stop training.')
        return

def test(model):
  test_X, test_batch_num = dataset.load_mnist(is_training=False)
  with tf.Session() as sess:
    test_acc = []
    test_loss = []
    for test_iter in range(test_batch_num):
      start = test_iter * batch_size
      end = start + batch_size
      # TODO: Evaluate the accuracy
      # TODO: print the output labels to a file

