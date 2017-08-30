'''
covlution layer，pool layer，initialization。。。。
'''
import tensorflow as tf
import numpy as np


# Weight initialization (Xavier's init)
def weight_xavier_init(shape, n_inputs, n_outputs, variable_name, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        initial = tf.random_uniform(shape, -init_range, init_range)
        return tf.Variable(initial, trainable=True, name=variable_name)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        initial = tf.truncated_normal(shape, stddev=stddev)
        return tf.Variable(initial, trainable=True, name=variable_name)


# Bias initialization
def bias_variable(shape, variable_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, trainable=True, name=variable_name)


# 2D convolution
def convolution_2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# Max Pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def batchnorm(Ylogits, n_out, is_test, convolutional=True):
    """
        Batch normalization on convolutional maps.
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            Ylogits:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            is_test: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
    # adding the iteration prevents from averaging across non-existing iterations
    beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                        name='gamma', trainable=True)
    exp_moving_avg = tf.train.ExponentialMovingAverage(decay=0.9)
    bnepsilon = 0.001
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, beta, gamma, bnepsilon)
    return Ybn


# Convert class labels from scalars to one-hot vectors
# 0 => [1 0 0 0 0 0 0 0 0 0]
# 1 => [0 1 0 0 0 0 0 0 0 0]
# 2 => [0 0 1 0 0 0 0 0 0 0]
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
