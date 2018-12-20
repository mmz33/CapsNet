import tensorflow as tf
from dataset import load_mnist
import logging
import numpy as np
from capsnet import CapsNet
import argparse
from config import get_from_config

train_log_dir = get_from_config('train_log_dir')
checkpoint_path = get_from_config('checkpoint_path')
batch_size = get_from_config('batch_size')

def train(model, restore_checkpoint=True):
  # loads the data
  train_X, train_Y, train_batch_num, val_X, val_Y, val_batch_num = load_mnist()

  num_of_epochs = get_from_config('epochs')

  # store the best loss so far which can be used in validation
  best_loss_val = np.inf
  with tf.Session() as sess:

    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
      saver.restore(sess, checkpoint_path)
    else:
      logging.info('Initializing variables...')
      sess.run(tf.global_variables_initializer())

    for epoch in range(num_of_epochs):
      logging.info('Start epoch %d' % (epoch+1))
      for train_iter in range(train_batch_num):

        start = train_iter * batch_size
        end = start + batch_size

        # Run the training operation and measure the loss
        _, global_step, train_loss, train_acc, _summary = \
          sess.run([model.train_op, model.global_step, model.total_loss,
                    model.accuracy, model.train_summary],
                    feed_dict={model.X: train_X[start:end],
                               model.Y: train_Y[start:end]})

        logging.info('global_step: {}'.format(global_step))
        logging.info('Epoch {}, Iteration {}/{}, loss: {}, accuracy: {}'
          .format(epoch+1, train_iter+1, train_batch_num, train_loss, train_acc))

        train_writer.add_summary(_summary, global_step)

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

        # train_writer.add_summary(_summary, global_step)
        # train_writer.flush()

        val_acc.append(train_acc)
        val_loss.append(train_loss)

      total_acc = np.mean(val_acc)
      total_loss = np.mean(val_loss)

      logging.info('Validation Epoch {}, Val accuracy: {}, Loss: {}'
        .format(epoch+1, total_acc, total_loss))

      if best_loss_val > total_loss:
        saver.save(sess, checkpoint_path, global_step=global_step)
        best_loss_val = total_loss
      else:
        logging.info('Error not improving. Stop training.')
        return

def predict(model):
  test_X = load_mnist(is_training=False)
  with tf.Session() as sess:
    test_loss = sess.run([model.total_loss], feed_dict={model.X: test_X})
    logging.info('test_loss: {}'.format(test_loss))
    with open('submission.csv', 'w') as out_file:
      out_file.write('ImageId,Label\n')
      for img_id, pred_out in enumerate(model.y_pred):
        out_file.write('%d,%d\n' % (img_id, pred_out))

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  parser = argparse.ArgumentParser()
  parser.add_argument('--run_train', type=bool, default=True,  help='Start training')
  parser.add_argument('--run_test',  type=bool, default=False, help='Start testing')
  FLAGS, unparsed = parser.parse_known_args()
  if FLAGS.run_train:
    logging.info('Start training...')
    model = CapsNet()
    train(model, restore_checkpoint=False)
    logging.info('End training...')
  elif FLAGS.run_test:
    logging.info('Start testing.')
    model = CapsNet(is_training=False)
    predict(model)
    logging.info('End testing.')