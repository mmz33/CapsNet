import tensorflow as tf


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


def squash(s_j, axis=-1, epsilon=1e-7, keepdims=True, name=None):
    """
    Non-linearity squashing function

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
    :param keepdims: A boolean, if True dimensions are not reduced
    :param name: A string, name of scope
    """
    with tf.name_scope(name, default_name='squash'):
        # compute the norm of the vector s
        squared_norm = tf.reduce_sum(tf.square(s_j), axis=axis, keepdims=keepdims)
        s_j_norm = tf.sqrt(squared_norm + epsilon)
        s_j_unit_scale = s_j / s_j_norm  # unit scaling (second part of the equation)
        add_scale = squared_norm / (1. + squared_norm)  # additional scaling (first part of the equation)
        v_j = s_j_unit_scale * add_scale  # element-wise product
    return v_j
