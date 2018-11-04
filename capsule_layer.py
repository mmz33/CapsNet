import tensorflow as tf
from config import get_from_config

class CapsuleLayer:
  """Represents a Capsule Layer

  Main paper ref: https://arxiv.org/abs/1710.09829

  Notations:
  - `i`       : index for low-level capsule i
  - `j`       : index for high-level capsule j
  - `u_i`     : activity vector of low-level capsule i
  - `u_j`     : activity vector of high-level capsule j
  - `W_ij`    : transformation weight matrix of low-level
                capsule i and high-level capsule j
  - `u_hat_ij`: transformed low-level capsule i activity vector
  - `b_ij`    : temporary values used in the routing algorithm
  - `c_ij`    : weight coupling coefficients
  - `s_j`     : linear combination of `c_ij` and `u_hat_ij`
  - `v_j`     : output activity vector of high-level capsule j --> [squash(s_j)]

  Referring to Hinton's paper, we have two additional layers namely: PrimaryCaps Layer
  and DigitCapsLayer. Now, PrimaryCaps layer is like a conv layer but with squash as
  non-linearity function. In addition, DigitCaps layer is fully connected with the
  previous layer (which is the PrimaryCaps layer in this case).

  Arguments:
    capsules_num: An Integer, the number of output capsules
    act_vec_len: An Integer, the length of the activity vector of a capsule

  Returns:
    4-D tensor representing the output of this layer
  """

  def __init__(self, capsules_num, act_vec_len):
    self.capsules_num = capsules_num
    self.act_vec_len = act_vec_len

  def __call__(self, inputs, kernel_size=None, strides=None):
    """ Here the logic of capsule layers is applied.

    The optional params `kernel_size` and `strides` are only needed
    for a `PrimaryCaps` Layer and so if they are both None, a
    `DigitCaps` Layer is applied. Otherwise, a `PrimaryCaps` Layer
    is applied.

    :param inputs: 4-D input tensor following `NHWC` format
      [batch, height, width, out_channels]
    :param kernel_size: 3-D tensor following `channel_last` format
      [height, width, out_channels]
    :param strides: 4-D tensor, [1, height, width, 1]
    :return: output tensors
    """

    if kernel_size is not None and strides is not None:
      # PrimaryCaps Layer

      self.kernel_size = kernel_size
      self.strides = strides

      # Apply the logic of a convolutional layer
      # Filters here are the number of capsules x length of the activity vector
      conv_out = tf.layers.conv2d(inputs=inputs,
                                  filters=self.capsules_num * self.act_vec_len,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  activation=tf.nn.relu)

      # Reshape capsules into one column vector to be fully connected with the next layer
      # New shape: [batch_size, capsules, vector_length]
      # Output shape: [batch_size, 1152, 8]
      capsules = tf.reshape(conv_out, shape=[inputs.shape[0], -1, self.act_vec_len])

      assert capsules.get_shape() == [inputs.shape[0], 1152, 8]

      # Apply squashing non-linearity function
      # Squash on the capsules vectors axis
      squashed_capsules = self.squash(capsules)

      # Returns tensor with shape: [batch_size, 1152, 8]
      return squashed_capsules

    elif kernel_size is None and strides is None:
      # DigitCapsLayer
      # Here we need to apply the routing mechanism
      return self.routing(inputs)
    else:
      raise ValueError('kernel_size and strides params should be either both None \
                       (for DigitCaps) or both not None (for PrimaryCaps)')

  def routing(self, inputs, num_out_caps=10, v_j_len=16):
    """ Dynamic Routing Algorithm

    :param inputs: A 3-D tensor, [batch_size, num_caps_l, act_len].
      The input from the previous layer l (PrimaryCaps Layer)
    :param num_out_caps: An integer, the number of output (high-level) capsules in layer l+1
    :param v_j_len: An integer, the length of the output (high-level) activity vector in layer l+1
    :param r: An integer, the number of routing iterations

    :return: A 4-D tensor, [batch_size, num_out_caps, v_j_len, 1].
      The activity vectors of the high-level capsules in layer l+1
    """

    batch_size = inputs.shape[0]
    num_caps_l = inputs.shape[1]

    # Create transformation weight matrix variable
    W_ij = tf.get_variable(name='Weight',
                           shape=[1, num_caps_l, num_out_caps, v_j_len, self.act_vec_len],
                           initializer=tf.random_normal_initializer(stddev=get_from_config('stddev')))

    # Replicate W_ij for batch_size copies
    # Output shape: [batch_size, 1152, 10, 16, 8]
    W_tiled = tf.tile(W_ij, [batch_size, 1, 1, 1, 1], name='W_tiled')

    assert W_tiled.get_shape().as_list() == [batch_size, 1152, 10, 16, 8]

    # Each previous layer output capsule is `act_vec_len` dimension
    # So, we need to expand the dimension by adding 1 at the end so that we
    # We can apply the matrix multiplication later with W_tiled later
    # Output shape: [batch_size, 1152, 8, 1]
    caps1_out_expand1 = tf.expand_dims(inputs, -1, name='caps1_out_expand1')

    assert caps1_out_expand1.get_shape().as_list() == [batch_size, 1152, 8, 1]

    # We also need to expand the dimension at axis 2 since we need to replicate (tf.tile)
    # for `num_out_caps` (10) copies to multiply with W_tiled
    # Output shape: [batch_size, 1152, 1, 8, 1]
    caps1_out_expand2 = tf.expand_dims(caps1_out_expand1, 2, name='caps1_out_expand2')

    assert caps1_out_expand2.get_shape().as_list() == [batch_size, 1152, 1, 8, 1]

    # Apply tf.tile
    # Output shape: [batch_size, 1152, 10, 8, 1]
    caps1_out_tiled = tf.tile(caps1_out_expand2, [1, 1, num_out_caps, 1, 1], 'caps1_out_tiled')

    assert caps1_out_tiled.get_shape().as_list() == [batch_size, 1152, 10, 8, 1]

    # This will result in a matrix of all predicated capsule vectors for all i and j
    # Each entry is u_hat_ij = W_ij x u_i
    # Output shape: [batch_size, 1152, 10, 16, 1]
    u_hat_vectors = tf.matmul(W_tiled, caps1_out_tiled, name='u_hat_vectors')

    assert u_hat_vectors.get_shape().as_list() == [batch_size, 1152, 10, 16, 1]

    # Line 2: create temporary weighting coefficients and initialize to zero

    # Output shape: [batch_size, 1152, 10, 1, 1]
    b_ij = tf.zeros(shape=[batch_size, num_caps_l, num_out_caps, 1, 1],
                    dtype=tf.float32,
                    name='temp_weighting_coeff')

    assert b_ij.get_shape().as_list() == [batch_size, 1152, 10, 1, 1]

    for r_iter in range(get_from_config('routing_iterations')):
      # Line 4: apply softmax on b_i for all capsule i in layer l

      # Output shape: [batch_size, 1152, 10, 1, 1]
      c_i = tf.nn.softmax(b_ij, axis=2, name='weighting_coefficients')

      assert c_i.get_shape().as_list() == [batch_size, 1152, 10, 1, 1]

      # Line 5: compute the weighted sum for all capsule j in layer l+1

      # Element-wise multiplication
      # Output shape: [batch_size, 1152, 10, 16, 1]
      weighted_pred = tf.multiply(c_i, u_hat_vectors, name='weighted_predictions')

      assert weighted_pred.get_shape().as_list() == [batch_size, 1152, 10, 16, 1]

      # Output shape: [batch_size, 1, 10, 16, 1]
      s_j  = tf.reduce_sum(weighted_pred,
                           axis=1,
                           keepdims=True,
                           name='weighed_sum')

      assert s_j.get_shape().as_list() == [batch_size, 1, 10, 16, 1]

      # Line 6: Apply squash function

      # Output shape: [batch_size, 1, 10, 16, 1]
      v_j = self.squash(s_j, axis=-2)

      assert v_j.get_shape().as_list() == [batch_size, 1, 10, 16, 1]

      # Line 7: update the b_ij variables to be used to update the c_ij variables

      # TODO: 1152 should not be hard coded
      # Output shape: [batch_size, 1152, 10, 16, 1]
      v_j_tiled = tf.tile(v_j, [1, 1152, 1, 1, 1], name='v_j_tiled')

      assert v_j_tiled.get_shape().as_list() == [batch_size, 1152, 10, 16, 1]

      # u_hat_ij * v_j
      # Output shape: [batch_size, 1152, 10, 16, 1]
      agreement = tf.matmul(u_hat_vectors, v_j_tiled, transpose_a=True, name='agreement')

      assert agreement.get_shape().as_list() == [batch_size, 1152, 10, 1, 1]

      b_ij = tf.add(b_ij, agreement, name='temp_weighting_coeff_update')

    return v_j

  @staticmethod
  def squash(s_j, axis=-1, epsilon=1e-7, name=None):
    """Non-linearity Squashing function

    The length of the capsule's activity vector represents the probability that the
    object it is detecting exists in the image and so we need to squash this vector
    to be in the range [0,1].

    Implementation ref: https://arxiv.org/abs/1710.09829

    Note that here we can't use tf.norm since the norm of the vector
    might be zero and so the training process will blow up with
    nans. Thus, we need to implement it manually by adding a tiny
    variable epsilon to tackle this problem.

    :param s_j: 3-D tensor, [batch_size, capsules, vector_length]
      we want to squash `vector_length` of the activity vector of capsule j
    :param axis: An integer, the axis on which squashing is applied
    :param name: A string, name of the function
    :return: v_j, squashed vector
    """

    with tf.name_scope(name, default_name='squash'):
      # compute the norm of the vector s
      squared_norm = tf.reduce_sum(tf.square(s_j),
                                   axis=axis,
                                   keepdims=True)

      s_j_norm = tf.sqrt(squared_norm + epsilon)

      # unit scaling (second part of the equation)
      s_j_unit_scale = s_j / s_j_norm

      # additional scaling (first part of the equation)
      add_scale = squared_norm / (1. + squared_norm)

      # element-wise product
      v_j = s_j_unit_scale * add_scale

    return v_j
