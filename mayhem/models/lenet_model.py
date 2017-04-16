#!/usr/bin/env python
"""
This file provides the definition of the LeNet model.
"""

import tensorflow as tf

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

class LeNetModel:
    """
    This LeNet model uses the well-known structure: two convolution layers and two
    fully-connected layers
    """
    @staticmethod
    def inference(x):
        with tf.name_scope('inference'):
            with tf.name_scope('conv1'):
                W_conv1 = tf.Variable(
                        tf.truncated_normal([5, 5, 1, 32], stddev=0.1),
                        name='weights')

                b_conv1 = tf.Variable(
                        tf.constant(0.1, shape=[32]),
                        name='bias')

                h_conv1 = tf.nn.relu(
                        tf.nn.conv2d(
                            tf.reshape(x, [-1, 28, 28, 1]),
                            W_conv1,
                            strides=[1, 1, 1, 1],
                            padding='SAME') + b_conv1)

            with tf.name_scope('pool1'):
                h_pool1 = tf.nn.max_pool(
                        h_conv1,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')

            with tf.name_scope('conv2'):
                W_conv2 = tf.Variable(
                        tf.truncated_normal([5, 5, 32, 64], stddev=0.1),
                        name='weights')

                b_conv2 = tf.Variable(
                        tf.constant(0.1, shape=[64]),
                        name='bias')

                h_conv2 = tf.nn.relu(
                        tf.nn.conv2d(
                            h_pool1,
                            W_conv2,
                            strides=[1, 1, 1, 1],
                            padding='SAME') + b_conv2)

            with tf.name_scope('pool2'):
                h_pool2 = tf.nn.max_pool(
                        h_conv2,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding='SAME')

            with tf.name_scope('fc1'):
                W_fc1 = tf.Variable(
                        tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1),
                        name='weights')

                b_fc1 = tf.Variable(
                        tf.constant(0.1, shape=[1024]),
                        name='bias')

                h_fc1 = tf.nn.relu(
                        tf.matmul(
                            tf.reshape(h_pool2, [-1, 7 * 7 * 64]),
                            W_fc1) + b_fc1)

            with tf.name_scope('fc2'):
                W_fc2 = tf.Variable(
                        tf.truncated_normal([1024, 10], stddev=0.1),
                        name='weights')

                b_fc2 = tf.Variable(
                        tf.constant(0.1, shape=[10]),
                        name='bias')

                h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

            logits = tf.identity(h_fc2, name='logits')

        return logits

    @staticmethod
    def loss(logits, labels):
        with tf.name_scope('loss'):
            labels = tf.to_int64(labels, name='labels')

            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels,
                    logits=logits,
                    name='cross_entropy')

            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')

            loss = tf.identity(cross_entropy_mean, name='loss')

        return loss 


    @staticmethod
    def train(loss, learning_rate):
        with tf.name_scope('train'):
            # Will be used in TensorBoard
            tf.summary.scalar('loss', loss)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate)

            global_step = tf.Variable(0, name='global_step', trainable=False)

            train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op


    @staticmethod
    def evaluation(logits, labels):
        with tf.name_scope('evaluation'):
            correct = tf.nn.in_top_k(logits, labels, 1)

            total_correct = tf.reduce_sum(
                    tf.cast(correct, tf.int32),
                    name='total_correct')

        return total_correct
            


def main(_):
    graph = tf.Graph()

    batch_size = 50
    learning_rate = 0.01

    with graph.as_default():
        x = tf.placeholder(tf.float32, [batch_size, 784])

        y_ = tf.placeholder(tf.int32, [batch_size])

        logits = LeNetModel.inference(x)

        loss = LeNetModel.loss(logits, y_)

        train_op = LeNetModel.train(loss, learning_rate)

        evaluation = LeNetModel.evaluation(logits, y_)

        graph_def = graph.as_graph_def()
        for node in graph_def.node:
            print(node.name)


if __name__ == '__main__':

    tf.app.run()
