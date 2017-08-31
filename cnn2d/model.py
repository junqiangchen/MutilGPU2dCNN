'''

'''
from cnn2d.layer import (convolution_2d, max_pool_2x2, weight_xavier_init, bias_variable)
import tensorflow as tf
import numpy as np


def create_conv_net(X, image_width, image_height, image_channel, image_labels, drop_conv, drop_hidden):
    # CNN model
    X1 = tf.reshape(X, [-1, image_width, image_height, image_channel])  # shape=(?, 32, 32, 3)
    # Layer 1
    W1 = weight_xavier_init(shape=[3, 3, 3, 32], n_inputs=3 * 3 * 3,
                            n_outputs=32, variable_name='conv1_W')  # 3x3x1 conv, 32 outputs,image shape[32,32]->[32,32]
    B1 = bias_variable([32], variable_name='conv1_B')

    conv1 = convolution_2d(X1, W1) + B1
    l1_conv = tf.nn.relu(conv1)  # shape=(?, 32, 32, 32)
    l1_drop = tf.nn.dropout(l1_conv, drop_conv)
    # Layer 2
    W2 = weight_xavier_init(shape=[3, 3, 32, 32], n_inputs=3 * 3 * 32,
                            n_outputs=32, variable_name='conv2_W')  # 3x3x1 conv, 32 outputs,image shape[32,32]->[16,16]
    B2 = bias_variable([32], variable_name='conv2_B')

    conv2 = convolution_2d(l1_drop, W2) + B2
    l2_conv = tf.nn.relu(conv2)  # shape=(?, 32, 32, 32)
    l2_pool = max_pool_2x2(l2_conv)  # shape=(?, 16, 16, 32)
    l2_drop = tf.nn.dropout(l2_pool, drop_conv)
    # Layer 3
    W3 = weight_xavier_init(shape=[3, 3, 32, 64], n_inputs=3 * 3 * 32,
                            n_outputs=64,
                            variable_name='conv3_W')  # 3x3x32 conv, 64 outputs,image shape[16,16]->[16,16]
    B3 = bias_variable([64], variable_name='conv3_B')

    conv3 = convolution_2d(l2_drop, W3) + B3
    l3_conv = tf.nn.relu(conv3)  # shape=(?, 16, 16, 64)
    l3_drop = tf.nn.dropout(l3_conv, drop_conv)
    # Layer 4
    W4 = weight_xavier_init(shape=[3, 3, 64, 64],
                            n_inputs=3 * 3 * 64,
                            n_outputs=64, variable_name='conv4_W')  # 3x3x64 conv, 64 outputs,image shape[16,16]->[8,8]
    B4 = bias_variable([64], variable_name='conv4_B')

    conv4 = convolution_2d(l3_drop, W4) + B4
    l4_conv = tf.nn.relu(conv4)  # shape=(?, 16, 16, 64)
    l4_pool = max_pool_2x2(l4_conv)  # shape=(?, 8, 8, 64)
    l4_drop = tf.nn.dropout(l4_pool, drop_conv)

    # Layer 5
    W5 = weight_xavier_init(shape=[3, 3, 64, 64], n_inputs=3 * 3 * 64,
                            n_outputs=64,
                            variable_name='conv5_W')  # 3x3x32 conv, 64 outputs,image shape[16,16]->[16,16]
    B5 = bias_variable([64], variable_name='conv5_B')

    conv5 = convolution_2d(l4_drop, W5) + B5
    l5_conv = tf.nn.relu(conv5)  # shape=(?, 16, 16, 64)
    l5_drop = tf.nn.dropout(l5_conv, drop_conv)
    # Layer 6
    W6 = weight_xavier_init(shape=[3, 3, 64, 64],
                            n_inputs=3 * 3 * 64,
                            n_outputs=64, variable_name='conv6_W')  # 3x3x64 conv, 64 outputs,image shape[16,16]->[8,8]
    B6 = bias_variable([64], variable_name='conv6_B')

    conv6 = convolution_2d(l5_drop, W6) + B6
    l6_conv = tf.nn.relu(conv6)  # shape=(?, 16, 16, 64)
    l6_pool = max_pool_2x2(l6_conv)  # shape=(?, 8, 8, 64)
    l6_drop = tf.nn.dropout(l6_pool, drop_conv)
    # Layer 7 - FC1
    W5_FC1 = weight_xavier_init(shape=[64 * 8 * 8, 512],
                                n_inputs=64 * 8 * 8, n_outputs=512,
                                variable_name='conv7_W')  # FC: 64x8x8 inputs, 512 outputs
    B5_FC1 = bias_variable([512], variable_name='conv7_B')

    l5_flat = tf.reshape(l6_drop, [-1, W5_FC1.get_shape().as_list()[0]])  # shape=(?, 512)
    FC1 = tf.matmul(l5_flat, W5_FC1) + B5_FC1
    l5_feed = tf.nn.relu(FC1)
    l5_drop = tf.nn.dropout(l5_feed, drop_hidden)
    # Layer 8 - FC2
    W6_FC2 = weight_xavier_init(shape=[512, image_labels],
                                n_inputs=512, n_outputs=image_labels,
                                variable_name='conv8_W')  # FC: 512 inputs, 2 outputs (labels)
    B6_FC2 = bias_variable([image_labels], variable_name='conv8_B')

    Y_pred = tf.nn.softmax(tf.matmul(l5_drop, W6_FC2) + B6_FC2)  # shape=(?, 2)

    return Y_pred


# Serve data by batches
def next_batch(train_images, train_labels, batch_size, index_in_epoch):
    start = index_in_epoch
    index_in_epoch += batch_size

    num_examples = train_images.shape[0]
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end], index_in_epoch


class cnn2dModule(object):
    """
    A cnn2d implementation

    :param image_height: number of height in the input image
    :param image_width: number of width in the input image
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param costname: name of the cost function.Default is "cross_entropy"
    """

    def __init__(self, image_height, image_width, channels=3, n_class=2, costname="cross_entropy"):
        self.n_class = n_class

        self.X = tf.placeholder("float", shape=[None, image_height * image_width * channels])
        self.Y_gt = tf.placeholder("float", shape=[None, n_class])
        self.lr = tf.placeholder('float')
        self.drop_conv = tf.placeholder('float')
        self.drop_hidden = tf.placeholder('float')

        Y_pred = create_conv_net(self.X, image_width, image_height, channels, n_class, self.drop_conv, self.drop_hidden)
        self.cost = self._get_cost(Y_pred, costname)

        correct_predict = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(self.Y_gt, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))
        self.predict = tf.argmax(Y_pred, 1)

    def _get_cost(self, Y_pred, cost_name):
        if cost_name == "cross_entropy":
            cost = -tf.reduce_sum(self.Y_gt * tf.log(Y_pred))
        return cost
    
    def _get_accuracy(self, Y_pred):
        correct_predict = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(self.Y_gt, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))
        return accuracy
    
    def train(self, train_images, train_lanbels, model_path, logs_path, learning_rate,
              dropout_conv=0.8, dropout_hidden=0.7, train_epochs=10000, batch_size=100):
        train_op = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.all_variables())

        tf.summary.scalar("loss", self.cost)
        tf.summary.scalar("accuracy", self.accuracy)
        merged_summary_op = tf.summary.merge_all()
        sess = tf.InteractiveSession()
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        sess.run(init)

        DISPLAY_STEP = 1
        index_in_epoch = 0

        for i in range(train_epochs):
            # get new batch
            batch_xs, batch_ys, index_in_epoch = next_batch(train_images, train_lanbels, batch_size, index_in_epoch)
            # check progress on every 1st,2nd,...,10th,20th,...,100th... step
            if i % DISPLAY_STEP == 0 or (i + 1) == train_epochs:
                train_accuracy = self.accuracy.eval(feed_dict={self.X: batch_xs[batch_size // 10:],
                                                               self.Y_gt: batch_ys[batch_size // 10:],
                                                               self.lr: learning_rate,
                                                               self.drop_conv: dropout_conv,
                                                               self.drop_hidden: dropout_hidden})
                validation_accuracy = self.accuracy.eval(feed_dict={self.X: batch_xs[0:batch_size // 10],
                                                                    self.Y_gt: batch_ys[0:batch_size // 10],
                                                                    self.lr: learning_rate,
                                                                    self.drop_conv: dropout_conv,
                                                                    self.drop_hidden: dropout_hidden})
                print('epochs %d training_accuracy / validation_accuracy => %.2f / %.2f ' % (
                    i, train_accuracy, validation_accuracy))

                if i % (DISPLAY_STEP * 10) == 0 and i:
                    DISPLAY_STEP *= 10

                    # train on batch
            _, summary = sess.run([train_op, merged_summary_op], feed_dict={self.X: batch_xs,
                                                                            self.Y_gt: batch_ys,
                                                                            self.lr: learning_rate,
                                                                            self.drop_conv: dropout_conv,
                                                                            self.drop_hidden: dropout_hidden})
            summary_writer.add_summary(summary, i)
        summary_writer.close()

        save_path = saver.save(sess, model_path)
        print("Model saved in file:", save_path)

    def prediction(self, model_path, test_images):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        sess.run(init)
        saver.restore(sess, model_path)

        predictvalue = np.zeros(test_images.shape[0])
        y_dummy = np.empty((test_images.shape[0], self.n_class))
        for i in range(0, test_images.shape[0]):
            predictvalue[i] = sess.run(self.predict, feed_dict={self.X: [test_images[i]],
                                                                self.Y_gt: y_dummy,
                                                                self.drop_conv: 0.8,
                                                                self.drop_hidden: 0.7})
        return predictvalue
