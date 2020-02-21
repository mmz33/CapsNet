import tensorflow.compat.v1 as tf
import numpy as np
from config import get_from_config
from capsnet import CapsNet
import os


class Engine:

    def __init__(self, datasets):
        """
        :param datasets: A namedTuple, representation of the dataset
        """
        self.datasets = datasets

    def init_engine(self, is_training=True):
        """
        This function initialize the engine from the config by extracting some parameters
        It also create a saver and the config proto for training configuration

        :param is_training:
        :return:
        """
        self.tf_log_dir = get_from_config('log')
        self.checkpoint_path = get_from_config('checkpoint_path')
        self.batch_size = get_from_config('batch_size')
        self.num_epochs = get_from_config('epochs')
        self.model = CapsNet(is_training=is_training)
        self.saver = None

        self.create_config_proto()

    def _create_saver(self):
        self.saver = tf.train.Saver()

    def create_config_proto(self):
        # see https://medium.com/@liyin2015/tensorflow-cpus-and-gpus-configuration-9c223436d4ef
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.9

    def train(self, restore_checkpoint):
        """
        Train CapsNet

        :param restore_checkpoint: A bool, if True the latest checkpoint is loaded
        """
        print('training...')
        train_X, train_Y = self.datasets.train
        val_X, val_Y = self.datasets.val
        save_filename = os.path.abspath(self.checkpoint_path)  # TF expects absolute path
        best_loss_val = np.inf
        with tf.Session(config=self.config) as sess:
            if not self.saver:
                self._create_saver()
            train_writer = tf.summary.FileWriter(self.tf_log_dir, sess.graph)
            # TODO: test
            if restore_checkpoint and tf.train.checkpoint_exists(self.checkpoint_path):
                print('restoring latest checkpoint')
                checkpoint_dir = tf.train.latest_checkpoint(os.path.dirname(self.checkpoint_path))
                self.saver.restore(sess, checkpoint_dir)
            else:
                print('initializing variables...')
                sess.run(tf.global_variables_initializer())

            num_train_samples = train_X.shape[0]
            num_train_batches = (num_train_samples + self.batch_size - 1) // self.batch_size

            num_val_samples = val_X.shape[0]
            num_val_batches = (num_val_samples + self.batch_size - 1) // self.batch_size

            for epoch in range(self.num_epochs):
                print('training epoch {}'.format(epoch + 1))
                for train_iter in range(num_train_batches):
                    start = train_iter * self.batch_size
                    end = min(num_train_samples, start + self.batch_size)

                    # Run the training operation and measure the loss
                    _, global_step, train_loss, train_acc, train_summary = sess.run(
                        [self.model.train_op,
                         self.model.global_step,
                         self.model.total_loss,
                         self.model.accuracy,
                         self.model.train_summary],
                        feed_dict={self.model.X: train_X[start:end],
                                   self.model.Y: train_Y[start:end],
                                   self.model.mask_with_labels: True})

                    print('batch {}/{}, loss: {}, accuracy: {:.2f}%'.format(
                        train_iter+1, num_train_batches, train_loss, train_acc))

                    train_writer.add_summary(train_summary, global_step)

                # Do validation at the end of each epoch
                print('start validation for epoch {}'.format(epoch + 1))
                val_acc = []
                val_loss = []
                for val_iter in range(num_val_batches):
                    start = val_iter * self.batch_size
                    end = start + self.batch_size

                    _, train_loss, train_acc, valid_summary = sess.run(
                        [self.model.train_op,
                         self.model.total_loss,
                         self.model.accuracy,
                         self.model.valid_summary],
                        feed_dict={self.model.X: val_X[start:end], self.model.Y: val_Y[start:end]})

                    print('batch {}/{}, loss: {}, accuracy: {:.2f}%'.format(
                        val_iter+1, num_val_batches, train_loss, train_acc))

                    train_writer.add_summary(valid_summary, global_step)

                    val_acc.append(train_acc)
                    val_loss.append(train_loss)

                train_writer.flush()
                total_acc = np.mean(val_acc)
                total_loss = np.mean(val_loss)
                print('val loss: {}, val accuracy: {:.2f}%'.format(total_loss, total_acc))

                if total_loss < best_loss_val:
                    chkpt_name= save_filename + '-epoch{}'.format(epoch + 1)
                    self.saver.save(sess=sess, save_path=chkpt_name, global_step=global_step)
                    best_loss_val = total_loss

    def test(self):
        """
        Test CapsNet
        """
        print('testing...')
        test_X, test_Y = self.datasets.test
        with tf.Session(config=self.config) as sess:
            if not self.saver:
                self._create_saver()
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            self.saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(checkpoint_dir))
            num_samples = test_X.shape[0]
            num_iters = (num_samples + self.batch_size - 1) // self.batch_size
            test_loss = []
            test_acc = []
            for test_iter in range(num_iters):
                start = test_iter * self.batch_size
                end = min(num_samples, start + self.batch_size)
                acc, loss = sess.run(
                    [self.model.accuracy, self.model.total_loss],
                    feed_dict={self.model.X: test_X[start:end], self.model.Y: test_Y[start:end]})
                print('batch {}/{}, loss: {}, accuracy: {:.2f}%'.format(test_iter, num_iters, loss, acc))
                test_acc += [acc]
                test_loss += [loss]
            print('Test accuracy: {}'.format(np.mean(test_acc)))
            print('Test loss: {}'.format(np.mean(test_loss)))

    def test_kaggle(self):
        """
        A simple wrapper to test kaggle digit recognizer task
        Task: https://www.kaggle.com/c/digit-recognizer/overview
        see also: run_kaggle.py
        """
        print('Run kaggle test...')
        test_X = self.datasets.test
        with tf.Session(config=self.config) as sess:
            if not self.saver:
                self._create_saver()
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            self.saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(checkpoint_dir))
            num_samples = test_X.shape[0]
            num_iters = (num_samples + self.batch_size - 1) // self.batch_size
            out_file = open('submission.csv', 'w')
            out_file.write('ImageId,Label\n')
            preds = []
            for test_iter in range(num_iters):
                start = test_iter * self.batch_size
                end = min(num_samples, start + self.batch_size)
                pred = sess.run(self.model.y_pred, feed_dict={self.model.X: test_X[start:end]})
                preds.extend(pred)
                print('Finished batch {}/{}'.format(test_iter + 1, num_iters))
            for i, pred in enumerate(preds):
                out_file.write(str(i + 1) + "," + str(pred) + '\n')
            out_file.close()
