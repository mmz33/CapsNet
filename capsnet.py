import tensorflow.compat.v1 as tf
from capsule_layer import CapsuleLayer
from config import get_from_config


class CapsNet:
    """
    Represents CapsNet Architecture

    In this class, the full architecture is built and the loss function is defined

    Implementation is based on the main paper `Dynamic Routing Between Capsules`
    Ref: https://arxiv.org/abs/1710.09829

    The CapsNet architecture is composed of two parts:
    1. Encoder (Convolutional layer, PrimaryCaps layer, and DigitCaps layer)
    2. Decoder (3 fully connected layers)
    """

    def __init__(self,
                 is_training=True,
                 height=28,
                 width=28,
                 channels=1,
                 primary_capsules=32,
                 activity_vector_len=8,
                 num_of_labels=10,
                 is_label_mask=False):

        """
        :param is_training: A boolean, if True this means we are in the training phase
        :param height: An integer, the height of the input image
        :param width: An integer, the width of the input image
        :param channels: An integer, the number of the input image channels
        :param primary_capsules: An integer, the number of primary capsules
        :param activity_vector_len: An integer, the length of activity vector
        :param num_of_labels: An integer, the number of output labels
        :param is_label_mask: A boolean, if True then the true labels are used for reconstruction else predicted labels
        """
        self.is_training = is_training
        self.height = height
        self.width = width
        self.channels = channels
        self.primary_capsules = primary_capsules
        self.activity_vector_len = activity_vector_len
        self.num_of_labels = num_of_labels
        self.mask_with_labels = tf.placeholder_with_default(is_label_mask, shape=(), name='mask_with_labels')
        self.batch_size = get_from_config('batch_size')

        self.X = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, height, width, channels], name='X')
        self.Y = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='Y')
        self.Y_enc = tf.one_hot(self.Y, depth=num_of_labels, axis=1, dtype=tf.int32, name='Y_enc')

        if is_training:
            self.build_capsnet()
            self.loss()
            self.compute_accuracy()
            self._summary()
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.optimizer = tf.train.AdamOptimizer()
            self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step, name='train_op')
        else:
            assert is_label_mask is False, \
                'During testing, predicted labels should be used for reconstruction and not the true labels.'

            self.build_capsnet()
            self.compute_accuracy()

    @staticmethod
    def safe_norm(s, axis=-1, epsilon=1e-7, keepdims=False, name=None):
        """
        Computes the norm of some input tensor

        :param s: A 4-D tf.Tensor
        :param axis: An integer, axis on which to apply safe_norm
        :param epsilon: A float, used to avoid dividing by zero
        :param keepdims: If true, retains reduced dimensions with length 1.
        :return: A tensor, the norm of tensor s
        """
        with tf.name_scope(name, default_name='safe_norm'):
            squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keepdims=keepdims)
            return tf.sqrt(squared_norm + epsilon)

    def build_capsnet(self):
        """
        Builds the CapsNet Architecture
        """

        # first convolutional layer
        conv1_kernel = tf.get_variable(
            "conv1_kernel", [9, 9, 1, 256], dtype=tf.float32, initializer=tf.random_normal_initializer(get_from_config('stddev')))
        conv1 = tf.nn.conv2d(input=self.X, filter=conv1_kernel, padding='VALID')  # (B,20,20,256)
        conv1 = tf.nn.relu(conv1)

        # PrimaryCaps layer
        with tf.variable_scope("primarycaps_layer"):
            primary_caps_layer = CapsuleLayer(num_capsules=self.primary_capsules, activity_vector_len=self.activity_vector_len)
            primary_caps_kernel = tf.get_variable("primary_caps_kernel", [9, 9, 256, self.activity_vector_len * self.primary_capsules],
                                                  dtype=tf.float32, initializer=tf.random_normal_initializer(get_from_config('stddev')))
            primary_capsules = primary_caps_layer(conv1, kernel=primary_caps_kernel, strides=2)  # (B,1152,8)

        # DigitCaps layer
        with tf.variable_scope("digitcaps_layer"):
            digit_caps_layer = CapsuleLayer(num_capsules=self.num_of_labels, activity_vector_len=self.activity_vector_len)
            digit_capsules = digit_caps_layer(primary_capsules)  # (B,1,10,16,1)
            digit_capsules = tf.squeeze(digit_capsules)  # (B,10,16)

        self.y_probs = self.safe_norm(digit_capsules, axis=-1, name='y_probs')  # (B,10)
        self.y_pred = tf.argmax(self.y_probs, axis=-1, name="y_pred", output_type=tf.int32)  # (B,)

        # For the reconstruction, we need to mask out all the output activity vectors except the longest one.
        # For that, we need to apply masking.
        with tf.variable_scope('masking'):
            reconstruct_targets = tf.cond(
                self.mask_with_labels, lambda: self.Y, lambda: self.y_pred, name='reconstruct_targets')  # (B,)

            reconstruct_mask = tf.one_hot(reconstruct_targets, depth=self.num_of_labels, name='reconstruct_mask')  # (B,10)
            reconstruct_mask_expanded = tf.expand_dims(reconstruct_mask, axis=-1, name='reconstruct_mask_expanded')  # (B,10,1)
            masked_output = tf.multiply(digit_capsules, reconstruct_mask_expanded, name='masked_output')  # (B,10,16)

            masked_output_flattened = tf.reshape(masked_output, shape=[self.batch_size, -1])

        with tf.variable_scope('fc_layers'):
            fc1 = tf.keras.layers.Dense(units=512, activation=tf.nn.relu, name='fc_layer1_out')
            fc1_out = fc1(masked_output_flattened)
            fc2 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu, name='fc_layer2_out')
            fc2_out = fc2(fc1_out)
            out = tf.keras.layers.Dense(
                units=self.height * self.width * self.channels, activation=tf.nn.sigmoid, name='fc_layer3_out')
            self.decoder_output = out(fc2_out)

    def loss(self):
        """
        Computes the total loss of the network

        Referring to the paper:
          total_loss = margin_loss + reconstruction_loss
        """

        # parameters needed to compute the total loss
        m_plus = get_from_config('m_plus')
        m_minus = get_from_config('m_minus')
        lambda_ = get_from_config('lambda')
        alpha = get_from_config('alpha')

        # margin_loss

        T_k = tf.cast(self.Y_enc, dtype=tf.float32)
        max_l = tf.square(tf.maximum(0., m_plus - self.y_probs), name='max_l') # (B,10)
        max_r = tf.square(tf.maximum(0., self.y_probs - m_minus), name='max_r') # (B,10)
        L = tf.add(T_k * max_l, lambda_ * (1 - T_k) * max_r, name='L') # (B,10)
        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=-1), name='margin_loss') # (B,)

        # reconstruction loss

        X_flat = tf.reshape(self.X, [-1, self.height * self.width * self.channels])
        # compute the squared difference between the original image and the reconstructed image
        squared_diff = tf.square(X_flat - self.decoder_output, name='squared_diff')
        self.reconstruct_loss = tf.reduce_mean(squared_diff, name='reconstruct_loss')

        self.total_loss = tf.add(self.margin_loss, alpha * self.reconstruct_loss, name='total_loss')

    def compute_accuracy(self):
        """
        Compute the prediction accuracy
        """

        correct_pred = tf.equal(self.Y, self.y_pred, name='correct_pred')
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy') * 100

    def _summary(self):
        """
        Creates training summaries and returns a merged train summary instance
        """

        # add scalar summaries
        train_summary = [tf.summary.scalar('train/margin_loss', self.margin_loss),
                         tf.summary.scalar('train/reconstruction_loss', self.reconstruct_loss),
                         tf.summary.scalar('train/total_loss', self.total_loss),
                         tf.summary.scalar('train/accuracy', self.accuracy)]

        # Add reconstructed image
        reconstructed_image = tf.reshape(
            self.decoder_output, shape=[self.batch_size, self.height, self.width, self.channels])
        train_summary.append(tf.summary.image('reconstructed_image', reconstructed_image))

        self.train_summary = tf.summary.merge(train_summary)
