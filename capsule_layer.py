import tensorflow.compat.v1 as tf
from config import get_from_config
from utils import squash


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

    def __init__(self,
                 num_primary_caps,
                 primary_caps_vec_len,
                 num_out_caps,
                 out_caps_vec_len):
        """
        :param num_primary_caps: An integer, the number of low-level capules
        :param primary_caps_vec_len: An integer, low-level capsule's activity vector's length
        :param num_out_caps: An integer, the number of high-level capules
        :param out_caps_vec_len: An integer, high-level capsule's activity vector's length
        """
        self.num_primary_caps = num_primary_caps
        self.primary_caps_vec_len = primary_caps_vec_len
        self.num_out_caps = num_out_caps
        self.out_caps_vec_len = out_caps_vec_len

    def __call__(self, inputs, kernel=None, strides=None):
        """
        Here the logic of capsule layers is applied.

        :param inputs: 4-D input tensor following `NHWC` format [batch, height, width, out_channels]
        :param kernel: 4-D tensor having shape: [height, width, in_channels, out_channels]
        :param strides: An integer, would be replicated for both H and W
        """
        if kernel is not None:
            # PrimaryCaps layer (low-level capsules)
            conv_out = tf.nn.conv2d(input=inputs, filter=kernel, strides=strides, padding='VALID', name='conv_out')
            conv_out = tf.nn.relu(conv_out)
            capsules = tf.reshape(
                conv_out, shape=[tf.shape(inputs)[0], -1, self.primary_caps_vec_len], name='capsules')  # (B,1152,8)
            squashed_capsules = squash(capsules)
            return squashed_capsules
        elif kernel is None and strides is None:
            # DigitCaps layer
            # Here we need to apply the routing mechanism
            return self.routing(inputs)
        else:
            raise ValueError('kernel_size and strides params should be either both None \
                       (for DigitCaps) or both not None (for PrimaryCaps)')

    def routing(self, inputs):
        """
        Dynamic Routing Algorithm

        :param inputs: A 3-D tensor, [batch_size, num_caps_l, act_len].
          The input from the previous layer l (PrimaryCaps Layer)
        :param num_out_caps: An integer, the number of output (high-level) capsules in layer l+1
        :param v_j_len: An integer, the length of the output (high-level) activity vector in layer l+1
        :return: A 4-D tensor, [batch_size, num_out_caps, v_j_len, 1].
          The activity vectors of the high-level capsules in layer l+1
        """
        v_j_len = self.out_caps_vec_len
        batch_size = tf.shape(inputs)[0]
        num_caps_l = 6 * 6 * self.num_primary_caps  # TODO: fix

        # create transformation weight matrix variable
        weight_init = tf.random_normal_initializer(stddev=get_from_config('stddev'))

        # shape: (1,1152,10,16,8)
        W_ij = tf.get_variable(
            name='W_ij',
            shape=[1, num_caps_l, self.num_out_caps, v_j_len, self.primary_caps_vec_len],
            initializer=weight_init
        )

        # create B copies of W_ij
        W_tiled = tf.tile(W_ij, [batch_size, 1, 1, 1, 1], name='W_tiled')  # (B,1152,10,16,8)

        caps1_out_expand1 = tf.expand_dims(inputs, -1, name='caps1_out_expand1')  # (B,1152,8,1)
        caps1_out_expand2 = tf.expand_dims(caps1_out_expand1, 2, name='caps1_out_expand2')  # (B,1152,1,8,1)
        caps1_out_tiled = tf.tile(
            caps1_out_expand2, [1, 1, self.num_out_caps, 1, 1], 'caps1_out_tiled')  # (B,1152,10,8,1)

        # each entry is u_hat_ij = W_ij x u_i
        u_hat_vectors = tf.matmul(W_tiled, caps1_out_tiled, name='u_hat_vectors')  # (B,1152,10,16,1)

        # create temporary weighting coefficients and initialize to zero

        b_ij = tf.zeros(
            shape=[batch_size, num_caps_l, self.num_out_caps, 1, 1], dtype=tf.float32, name='b_ij')  # (B,1152,10,1,1)

        rounds = get_from_config('routing_iterations')

        v_j = None
        for r_iter in range(rounds):
            c_i = tf.nn.softmax(b_ij, axis=2, name='weighting_coefficients')  # (B,1152,10,1,1)
            weighted_pred = tf.multiply(c_i, u_hat_vectors, name='weighted_pred')  # (B,1152,10,16,1)
            s_j = tf.reduce_sum(weighted_pred, axis=1, keepdims=True, name='s_j')  # (B,1,10,16,1)
            v_j = squash(s_j, axis=-2)  # (B,1,10,16,1)
            v_j_tiled = tf.tile(v_j, [1, num_caps_l, 1, 1, 1], name='v_j_tiled')  # (B,1152,10,16,1)
            agreement = tf.matmul(u_hat_vectors, v_j_tiled, transpose_a=True, name='agreement')  # (B,1152,10,1,1)
            b_ij = tf.add(b_ij, agreement, name='temp_weighting_coeff_update')
        return v_j

