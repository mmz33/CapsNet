import tensorflow.compat.v1 as tf
from config import get_from_config


class CapsuleLayer:
    """
    Represents a Capsule Layer

    Paper: "Dynamic Routing Between Capsules"
    ref: https://arxiv.org/abs/1710.09829

    Notation (refer to paper):
    - i        : index for low-level capsule i
    - j        : index for high-level capsule j
    - u_i      : activity vector of low-level capsule i
    - u_j      : activity vector of high-level capsule j
    - W_ij     : transformation weight matrix of low-level capsule i and high-level capsule j
    - u_hat_ij : transformed low-level capsule i activity vector
    - b_ij     : temporary values used in the routing algorithm
    - c_ij     : weight coupling coefficients
    - s_j      : weighted linear combination of "c_ij" and "u_hat_ij"
    - v_j      : output activity vector of high-level capsule j {squash(s_j)}
    """

    def __init__(self, num_capsules, activity_vector_len):
        """
        :param num_capsules: An integer, the number of capsules
        :param activity_vector_len: An integer, the length of the activity vector
        """
        self.num_capsules = num_capsules
        self.activity_vector_len = activity_vector_len

    def __call__(self, inputs, kernel=None, strides=None):
        """Here the logic of capsule layers is applied.

        :param inputs: 4-D input tensor following `NHWC` format [batch, height, width, out_channels]
        :param kernel: 4-D tensor having shape: [height, width, in_channels, out_channels]
        :param strides: An integer, would be replicated for both H and W
        """
        if kernel is not None and strides is not None:
            # PrimaryCaps layer
            conv_out = tf.nn.conv2d(input=inputs, filter=kernel, strides=strides, padding='VALID', name='conv_out')
            conv_out = tf.nn.relu(conv_out)
            capsules = tf.reshape(
                conv_out, shape=[inputs.shape[0], -1, self.activity_vector_len], name='capsules') # (B,1152,8)
            squashed_capsules = self.squash(capsules)
            return squashed_capsules
        elif kernel is None and strides is None:
            # DigitCaps layer
            # Here we need to apply the routing mechanism
            return self.routing(inputs, self.num_capsules)
        else:
            raise ValueError('kernel_size and strides params should be either both None \
                       (for DigitCaps) or both not None (for PrimaryCaps)')

    def routing(self, inputs, num_out_caps, v_j_len=16):
        """
        Dynamic Routing Algorithm

        :param inputs: A 3-D tensor, [batch_size, num_caps_l, act_len].
          The input from the previous layer l (PrimaryCaps Layer)
        :param num_out_caps: An integer, the number of output (high-level) capsules in layer l+1
        :param v_j_len: An integer, the length of the output (high-level) activity vector in layer l+1
        :return: A 4-D tensor, [batch_size, num_out_caps, v_j_len, 1].
          The activity vectors of the high-level capsules in layer l+1
        """
        inputs_shape = inputs.get_shape().as_list()
        batch_size = inputs_shape[0]
        num_caps_l = inputs_shape[1]

        # create transformation weight matrix variable
        weight_init = tf.random_normal_initializer(stddev=get_from_config('stddev'))
        # shape: (1,1152,10,16,8)
        W_ij = tf.get_variable(
            name='W_ij', shape=[1, num_caps_l, num_out_caps, v_j_len, self.activity_vector_len], initializer=weight_init)

        # create B copies of W_ij
        W_tiled = tf.tile(W_ij, [batch_size, 1, 1, 1, 1], name='W_tiled') # (B,1152,10,16,8)

        caps1_out_expand1 = tf.expand_dims(inputs, -1, name='caps1_out_expand1') #  (B,1152,8,1)
        caps1_out_expand2 = tf.expand_dims(caps1_out_expand1, 2, name='caps1_out_expand2') # (B,1152,1,8,1)
        caps1_out_tiled = tf.tile(caps1_out_expand2, [1, 1, num_out_caps, 1, 1], 'caps1_out_tiled') # (B,1152,10,8,1)

        # each entry is u_hat_ij = W_ij x u_i
        u_hat_vectors = tf.matmul(W_tiled, caps1_out_tiled, name='u_hat_vectors') # (B,1152,10,16,1)

        # create temporary weighting coefficients and initialize to zero
        b_ij = tf.zeros(shape=[batch_size, num_caps_l, num_out_caps, 1, 1], dtype=tf.float32, name='b_ij') # (B,1152,10,1,1)

        rounds = get_from_config('routing_iterations')

        # TODO: use tf.while_loop
        v_j = None
        for r_iter in range(rounds):
            c_i = tf.nn.softmax(b_ij, axis=2, name='weighting_coefficients') # (B,1152,10,1,1)
            weighted_pred = tf.multiply(c_i, u_hat_vectors, name='weighted_pred') # (B,1152,10,16,1)
            s_j = tf.reduce_sum(weighted_pred, axis=1, keepdims=True, name='s_j') # (B,1,10,16,1)
            v_j = self.squash(s_j, axis=-2) # (B,1,10,16,1)
            v_j_tiled = tf.tile(v_j, [1, num_caps_l, 1, 1, 1], name='v_j_tiled') # (B,1152,10,16,1)
            agreement = tf.matmul(u_hat_vectors, v_j_tiled, transpose_a=True, name='agreement') # (B,1152,10,1,1)
            b_ij = tf.add(b_ij, agreement, name='temp_weighting_coeff_update')
        return v_j

    @staticmethod
    def squash(s_j, axis=-1, epsilon=1e-7, keepdims=True, name=None):
        """Non-linearity squashing function

        The length of the capsule's activity vector represents the probability that the
        object it is detecting exists in the image and so we need to squash this vector
        to be in the range [0,1].

        Implementation ref: https://arxiv.org/abs/1710.09829

        Note that here we can't use tf.norm since the norm of the vector
        might be zero and so the training process will blow up with
        nans. Thus, we need to implement it manually in order to add an epsilon
        to tackle this problem.

        :param s_j: N-D tensor where the axis number of `axis` correspond to the activity vector dimension
        :param axis: An integer, the axis on which squashing is applied
        :param epsilon: A float, small number to avoid dividing by zero
        :param keepdims:
        :param name:
        """
        with tf.name_scope(name, default_name='squash'):
            squared_norm = tf.reduce_sum(tf.square(s_j), axis=axis, keepdims=keepdims) # compute the norm of the vector s
            s_j_norm = tf.sqrt(squared_norm + epsilon)
            s_j_unit_scale = s_j / s_j_norm # unit scaling (second part of the equation)
            add_scale = squared_norm / (1. + squared_norm) # additional scaling (first part of the equation)
            v_j = s_j_unit_scale * add_scale # element-wise product
        return v_j
