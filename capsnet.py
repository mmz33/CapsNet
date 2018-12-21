import tensorflow as tf
from capsule_layer import CapsuleLayer
from config import get_from_config

class CapsNet:
  """Represents CapsNet Architecture

  In this class, the full architecture is built and the loss function is defined

  Implementation is based on the main paper `Dynamic Routing Between Capsules`
  Ref: https://arxiv.org/abs/1710.09829

  The CapsNet architecture is composed of two parts:
  1. Encoder (Conv layer, PrimaryCaps layer, and DigitCaps layer)
  2. Decoder (3 fully connected layers)

  Arguments:
    is_training: A boolean, if True this means we are in the training phase
    height: An integer, the height of the input image
    width: An integer, the width of the input image
    channels: An integer, the number of the input image channels
    num_of_labels: An integer, the number of output labels
    mask_with_labels: A boolean, if True then the true labels are used
    for reconstruction otherwise the predicted labels are used
  """

  def __init__(self,
               is_training=True,
               height=28,
               width=28,
               channels=1,
               num_of_labels=10,
               is_label_mask=False):

    self.is_training = is_training
    self.height = height
    self.width = width
    self.channels = channels
    self.num_of_labels = num_of_labels

    self.batch_size = get_from_config('batch_size')
    assert self.batch_size is not None

    # self.graph = tf.Graph()
    #
    # with self.graph.as_default():
    self.mask_with_labels = tf.placeholder_with_default(is_label_mask,
                                                        shape=(),
                                                        name="mask_with_labels")

    if is_training:
      self.X = tf.placeholder(dtype=tf.float32,
                              shape=(self.batch_size, height, width, channels),
                              name='X')

      self.Y = tf.placeholder(dtype=tf.int64, shape=(self.batch_size,), name='Y')
      self.Y_enc = tf.one_hot(self.Y, depth=num_of_labels, axis=1, dtype=tf.int64, name='Y_enc')

      self.build_capsnet()
      self.loss()
      self._summary()
      self.global_step = tf.Variable(0, trainable=False, name='global_step')
      self.optimizer = tf.train.AdamOptimizer()
      self.train_op = self.optimizer.minimize(self.total_loss,
                                              global_step=self.global_step,
                                              name='train_op')
    else:
      assert is_label_mask == False, \
        'During testing, predicted labels should be used for reconstruction and not the true labels.'

      self.X = tf.placeholder(dtype=tf.float32,
                              shape=(self.batch_size, height, width, channels),
                              name='test_X')
      self.build_capsnet()

  @staticmethod
  def safe_norm(s, axis=-1, epsilon=1e-7, keepdims=False, name=None):
    """Computes the norm of s avoiding nan problem (0 norm)

    It uses epsilon to make sure tf.sqrt(squared_norm + epsilon)
    will not output nan when squared_norm is 0

    :param s: A 4-D tf.Tensor
    :param axis: An integer, axis on which to apply safe_norm
    :param epsilon: A float, used to avoid nan problem
    :param keepdims: If true, retains reduced dimensions with length 1.
    :param name: A string, the name of the scope
    :return: A tensor, norm of tensor s
    """

    with tf.name_scope(name, default_name='safe_norm'):
      squared_norm = tf.reduce_sum(tf.square(s),
                                   axis=axis,
                                   keepdims=keepdims)

      return tf.sqrt(squared_norm + epsilon)

  def build_capsnet(self):
    """Builds the CapsNet Architecture"""

    # 1.Encoder

    # Convolutional Layer
    # Return tensor with shape [batch_size, 20, 20, 256]
    with tf.variable_scope('Conv1_layer'):
      conv1 = tf.layers.conv2d(inputs=self.X,
                               filters=256,
                               kernel_size=9,
                               padding='VALID')

    # PrimaryCaps Layer
    # Return tensor with shape [batch_size, 1152, 8]
    with tf.variable_scope('PrimaryCaps_layer'):
      primary_caps_layer = CapsuleLayer(capsules_num=32, act_vec_len=8)
      primary_capsules = primary_caps_layer(conv1, kernel_size=9, strides=2)

    # DigitCaps Layer
    # Return tensor with shape [batch_size, 1, 10, 16, 1]
    with tf.variable_scope('DigitCaps_layer'):
      digit_caps_layer = CapsuleLayer(capsules_num=self.num_of_labels,
                                      act_vec_len=8)
      digit_capsules = digit_caps_layer(primary_capsules)

      assert digit_capsules.get_shape().as_list() == \
             [self.batch_size, 1, self.num_of_labels, 16, 1]

      digit_capsules = tf.reshape(digit_capsules, shape=[self.batch_size, -1, 16, 1])

      assert digit_capsules.get_shape().as_list() == \
             [self.batch_size, 10, 16, 1]

    # 2.Decoder

    # Compute the output probabilities for each class
    # From [batch_size, 10, 16, 1] to [batch_size, 10, 1]
    self.y_probs = self.safe_norm(digit_capsules, axis=-2, name='y_probs')

    assert self.y_probs.get_shape().as_list() == [self.batch_size, 10, 1]

    # This is the index of the vector (class) having the largest length
    # Output shape: [batch_size, 1, 1]
    y_probs_argmax = tf.argmax(self.y_probs, axis=1, name="y_probs_argmax")

    assert y_probs_argmax.get_shape().as_list() == [self.batch_size, 1]

    # (batch_size,)
    # or reshape with shape=(batch_size,)
    self.y_pred = tf.squeeze(y_probs_argmax, axis=[1], name="y_pred")

    assert self.y_pred.shape == (self.batch_size,)

    # For the reconstruction, we need to mask out all the output
    # activity vectors except the longest one. For that, we need
    # to apply masking.

    with tf.variable_scope('masking'):
      # Return:
      # - y       if mask_with_labels is True
      # - y_pred  otherwise
      reconst_targets = tf.cond(self.mask_with_labels,
                                lambda: self.Y,
                                lambda: self.y_pred,
                                name='reconst_targets')

      # Create reconstruction mask
      # Output shape: [batch_size, 10]
      reconst_mask = tf.one_hot(reconst_targets,
                                depth=self.num_of_labels,
                                name='reconst_mask')

      # The shape of digit_capsules is [batch_size, 10, 16, 1]
      # Then, we need to reshape the reconstruction mask to apply it
      reconst_mask_reshaped = tf.reshape(reconst_mask,
                                         shape=[self.batch_size, self.num_of_labels, 1, 1],
                                         name='reconst_mask_reshaped')

      # Apply mask
      # Output shape: [batch_size, 10, 16, 1]
      masked_output = tf.multiply(digit_capsules,
                                  reconst_mask_reshaped,
                                  name='masked_output')

      assert masked_output.get_shape().as_list() == [self.batch_size, 10, 16, 1]

      # Flatten the masked output vector to be used in the fully connected layers
      masked_output_flattened = tf.reshape(masked_output,
                                           shape=[self.batch_size, -1])

      assert masked_output_flattened.get_shape().as_list() == [self.batch_size, 160]

      with tf.variable_scope('FC_layer'):
        fc_layer1_out = tf.layers.dense(masked_output_flattened,
                                        units=512,
                                        activation=tf.nn.relu,
                                        name='fc_layer1_out')

        fc_layer2_out = tf.layers.dense(fc_layer1_out,
                                        units=1024,
                                        activation=tf.nn.relu,
                                        name='fc_layer2_out')

        self.decoder_output = tf.layers.dense(fc_layer2_out,
                                        units=self.height*self.width*self.channels,
                                        activation=tf.nn.sigmoid,
                                        name='fc_layer3_out')

  def loss(self):
    """Computes the total loss of the network

    Referring to the original paper:
      total_loss = margin_loss + reconstruction_loss
    """

    # 1. margin_loss

    m_plus = get_from_config('m_plus')
    m_minus = get_from_config('m_minus')
    lambda_ = get_from_config('lambda')
    alpha = get_from_config('alpha')

    T_k = tf.to_float(self.Y_enc)

    max_l = tf.square(tf.maximum(0., m_plus - self.y_probs),
                      name='max_l')

    assert max_l.get_shape().as_list() == [self.batch_size, self.num_of_labels, 1]

    correct_label_loss = tf.reshape(max_l,
                                    shape=[-1, self.num_of_labels],
                                    name='correct_label_loss')

    max_r = tf.square(tf.maximum(0., self.y_probs - m_minus),
                      name='max_r')

    assert max_r.get_shape().as_list() == [self.batch_size, self.num_of_labels, 1]

    incorrect_label_loss = tf.reshape(max_r,
                                      shape=[-1, self.num_of_labels],
                                      name='incorrect_label_loss')

    L = tf.add(T_k * correct_label_loss,
               lambda_ * (1 - T_k) * incorrect_label_loss,
               name='L')

    self.margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=-1), name='margin_loss')

    # 2. reconstruction loss

    # Reshape original image
    X_flat = tf.reshape(self.X, [-1, self.height * self.width * self.channels])

    # Compute the squared difference between the original image and the
    # reconstructed image
    squared_diff = tf.square(X_flat - self.decoder_output,
                             name='squared_diff')

    self.reconst_loss = tf.reduce_mean(squared_diff, name='reconst_loss')

    self.total_loss = tf.add(self.margin_loss, alpha * self.reconst_loss, name='total_loss')

  def compute_accuracy(self):
    """Compute the prediction accuracy"""
    correct_pred = tf.equal(self.Y, self.y_pred, name='correct_pred')
    self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy') * 100

  def _summary(self):
    """Merges training summaries"""

    train_summary = []

    # Add the loss values to the summary
    train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
    train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconst_loss))
    train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))

    # Add reconstructed image
    reconst_image = tf.reshape(self.decoder_output,
                               shape=(self.batch_size, self.height, self.width, self.channels))
    train_summary.append(tf.summary.image('reconstructed_image', reconst_image))

    self.compute_accuracy()
    train_summary.append(tf.summary.scalar('train/accuracy', self.accuracy))

    # Merge all the summaries
    self.train_summary = tf.summary.merge(train_summary)

