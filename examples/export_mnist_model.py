#!/usr/bin/env python

"""
Export trained MNIST model to a .pb file.
"""

import os
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

flags = flags_lib
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "", """Directory to store model files""")
flags.DEFINE_boolean("train", False, """Whether to retrain the model""")

def train_and_save_model(model_dir):
  mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

  with tf.Graph().as_default():
    # Declare placeholders
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Declare variables 
    with tf.name_scope('conv1'):
      W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name='weights')
      b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name='bias')

    with tf.name_scope('conv2'):
      W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1), name='weights')
      b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]), name='bias')

    with tf.name_scope('fc1'):
      W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1), name='weights')
      b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='bias')

    with tf.name_scope('fc2'):
      W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name='weights')
      b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]), name='bias')
    
    # Declare computation nodes
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(
        tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    h_conv2 = tf.nn.relu(
        tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    h_fc1 = tf.nn.relu(tf.matmul(tf.reshape(h_pool2, [-1, 7 * 7 * 64]), W_fc1) + b_fc1)

    y = tf.identity(tf.matmul(h_fc1, W_fc2) + b_fc2, name='logits')

    # Declare loss related nodes
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())

      for i in range(2000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
          print('Model saved in file %s' % saver.save(sess, model_dir + '/model.chkp'))
        sess.run(train_op, feed_dict={ x: batch[0], y_: batch[1] })

      print("Test accuracy: %g" % sess.run(accuracy, feed_dict={ x: mnist.test.images, y_: mnist.test.labels }))

def load_and_freeze_graph(model_dir):
  """
  Reference: https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc
  """
  checkpoint = tf.train.get_checkpoint_state(model_dir)
  saver = tf.train.import_meta_graph(checkpoint.model_checkpoint_path + '.meta', clear_devices=True)
  graph = tf.get_default_graph()
  input_graph_def = graph.as_graph_def()

  with tf.Session() as sess:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, ['logits'])
    with gfile.GFile(model_dir + '/model.pb', 'wb') as f:
      f.write(output_graph_def.SerializeToString())


def main(_):
  if not os.path.isdir(FLAGS.model_dir):
    os.mkdir(FLAGS.model_dir)

  if FLAGS.train:
    train_and_save_model(FLAGS.model_dir) 

  load_and_freeze_graph(FLAGS.model_dir)

if __name__ == '__main__':
  tf.app.run()
