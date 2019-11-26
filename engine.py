import tensorflow.compat.v1 as tf
import numpy as np
from capsnet import CapsNet
import argparse
from config import get_from_config
from dataset import load_mnist

train_log_dir = get_from_config('train_log_dir')
checkpoint_path = get_from_config('checkpoint_path')
batch_size = get_from_config('batch_size')

def train(restore_checkpoint=True):
  num_of_epochs = get_from_config('epochs')

  # store the best loss so far which can be used in validation
  best_loss_val = np.inf
  with tf.Session() as sess:
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
      saver.restore(sess, checkpoint_path)
    else:
      print('Initializing variables...')
      sess.run(tf.global_variables_initializer())

    num_train_samples = train_X.shape[0]
    num_train_batches = num_train_samples // batch_size

    num_val_samples = val_X.shape[0]
    num_val_batches = num_val_samples // batch_size

    for epoch in range(num_of_epochs):
      print('start epoch %d' % (epoch+1))

      for train_iter in range(num_train_batches):

        start = train_iter * batch_size
        end = min(num_train_samples, start + batch_size)

        # Run the training operation and measure the loss
        _, global_step, train_loss, train_acc, _summary = sess.run(
          [model.train_op, model.global_step, model.total_loss, model.accuracy, model.train_summary],
          feed_dict={model.X: train_X[start:end], model.Y: train_Y[start:end]})

        print('batch {}/{}, loss: {}, accuracy: {:.2f}%'
              .format(train_iter+1, num_train_batches, train_loss, train_acc))

        train_writer.add_summary(_summary, global_step)

      # Do validation at the end of each epoch
      print('start validation for epoch {}'.format(epoch))
      val_acc = []
      val_loss = []
      for val_iter in range(num_val_batches):
        start = val_iter * batch_size
        end = start + batch_size

        _, global_step, train_loss, train_acc, _summary = sess.run(
          [model.train_op, model.global_step, model.total_loss, model.accuracy, model.train_summary],
          feed_dict={model.X: val_X[start:end], model.Y: val_Y[start:end]})

        print('batch {}/{}, loss: {}, accuracy: {:.2f}%'.format(val_iter+1, num_val_batches, train_loss, train_acc))

        train_writer.add_summary(_summary, global_step)

        val_acc.append(train_acc)
        val_loss.append(train_loss)

      train_writer.flush()

      total_acc = np.mean(val_acc)
      total_loss = np.mean(val_loss)

      print('total val loss: {}, total val accuracy: {:.2f}%'.format(total_acc, total_loss))

      if best_loss_val > total_loss:
        saver.save(sess, checkpoint_path, global_step=global_step)
        best_loss_val = total_loss
      else:
        print('Error not improving. Stop training.')
        return

def predict(ckpt_num):
  # TODO: restore model first and then predict
  with tf.Session() as sess:
    test_loss = sess.run(model.total_loss, feed_dict={model.X: test_X})
    print('test_loss: {}'.format(test_loss))
    with open('submission.csv', 'w') as out_file:
      out_file.write('ImageId,Label\n')
      for img_id, pred_out in enumerate(model.y_pred):
        out_file.write('%d,%d\n' % (img_id, pred_out))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--train', action="store_true", help='Start training')
  parser.add_argument('--test',  action="store_true", help='Start testing')
  FLAGS = parser.parse_args()

  train_X, train_Y, val_X, val_Y, test_X, test_Y = load_mnist()

  if FLAGS.train:
    print('Start training...')
    model = CapsNet()
    train(restore_checkpoint=True)
    print('End training...')
  elif FLAGS.test:
    print('Start testing.')
    model = CapsNet(is_training=False)
    predict(1)
    print('End testing.')