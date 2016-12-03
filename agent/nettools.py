import tensorflow as tf


def weight_variable(shape, stddev=0.1):
    """
    Create weights used in neural networks by sampling from a truncated
    normal distribution.
    :param list[int] shape: The dimensions of the weight variable to create.
    :param dbl stddev: The standard deviation of the normal distribution to
        sample the weights from. Defaults to 0.1.
    :return: A weight matrix.
    :rtype: tf.Tensor.
    """
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, value=0.1):
    """
    Create the bias used in neural networks as a constant shift.
    :param list[int] shape: The dimensions of the bias variable to create.
    :param dbl value: The bias to create. Defaults to 0.1.
    :return: A bias matrix.
    :rtype: tf.Tensor.
    """
    initial = tf.constant(value=value, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """
    Create the 2d convolution of an object x with a weight matrix W.
    :param tf.Tensor x: The left component of the convolution.
    :param tf.Tensor W: The right component of the convolution. This is the
        moving filter.
    :return: The convolved object.
    :rtype: tf.Tensor.
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    Do a 2x2 max pooling.
    :param tf.Tensor x: The input for pooling.
    :return: 2x2 max pooling applied to the input.
    :rtype: tf.Tensor.
    """
    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')