import functools
import operator
import network.nettools as nt
import tensorflow as tf


def deepmind_convnet(input_layer_shape,
                     output_layer_shape,
                     device='/cpu:0'):
    """
    From arXiv:1312.5602
    The input to the neural network consists of an 84 x 84 x 4 image produced
    by the pre-processng map phi. The first hidden layer convolves 32 filters
    of 8 x 8 with stride 4 with the input image and applies a rectifier
    nonlinearity. The second hidden layer convolves 64 filters of 4 x 4 with
    stride 2, again followed by a rectifier nonlinearity. This is followed by a
    third convolutional layer that convolves 64 filters of 3 x 3 with stride 1
    followed by a rectifier. The final hidden layer is fully-connected and
    consists of 512 rectifier units. The output layer is a fully-connected
    linear layer with a single output for each valid action. The number of
    valid actions varied between 4 and 18 on the games we considered.
    """
    with tf.device(device_name_or_function=device):
        # Placeholders for feed and output
        s = tf.placeholder(tf.float32, shape=input_layer_shape)
        q = tf.placeholder(tf.float32, shape=[None])

        # First convolutional layer
        # The first hidden layer convolves 32 filters of 8 x 8 with stride 4
        # with the input image and applies a rectifier nonlinearity
        W_conv1 = nt.weight_variable([8, 8, 4, 32])
        b_conv1 = nt.bias_variable([32])
        h_conv1 = tf.nn.relu(nt.conv2d(s, W_conv1) + b_conv1)

        # Second convolutional layer
        # The second hidden layer convolves 64 filters of 4 x 4 with
        # stride 2, again followed by a rectifier nonlinearity.
        W_conv2 = nt.weight_variable([4, 4, 2, 64])
        b_conv2 = nt.bias_variable([64])
        h_conv2 = tf.nn.relu(nt.conv2d(h_conv1, W_conv2) + b_conv2)

        # Third convolutional layer
        # This is followed by a third convolutional layer that convolves 64
        # filters of 3 x 3 with stride 1 followed by a rectifier.
        W_conv3 = nt.weight_variable([3, 3, 1, 64])
        b_conv3 = nt.bias_variable([64])
        h_conv3 = tf.nn.relu(nt.conv2d(h_conv2, W_conv3) + b_conv3)

        # The final hidden layer is fully-connected and
        # consists of 512 rectifier units.
        h_conv_size = functools.reduce(
            operator.mul,
            filter(None, h_conv3.get_shape().as_list()),
            1)
        h_conv3_flat = tf.reshape(h_conv3, [-1, h_conv_size])
        W_fc1 = nt.weight_variable([h_conv_size, 512])
        b_fc1 = nt.bias_variable([512])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # Dropout layer to prevent overfitting (not in original paper, keep as
        # 1 to get same results
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # The output layer is a fully-connected linear layer with a single
        # output for each valid action
        W_fc2 = nt.weight_variable([512] + output_layer_shape[1:])
        b_fc2 = nt.bias_variable(output_layer_shape[1:])
        q_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return s, q, q_conv, keep_prob
