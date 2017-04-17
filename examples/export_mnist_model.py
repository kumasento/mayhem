#!/usr/bin/env python

"""
Export trained MNIST model to a .pb file.

The core idea in this snippet:
  1. A saver will be configured while training the model, which will 
  save the model at some checkpoints.
  2. There will be a meta file stored in the checkpoint directory that
  the saver creates, which will be used to restore the state of the 
  last saved checkpoint.
  3. The model graph can be saved without reloading - just create the 
  graph definition by using:

    output_graph_def = graph_util.convert_variables_to_constants(
        sess, input_graph_def, ['logits'])
    with gfile.GFile(model_dir + '/model.pb', 'wb') as f:
      f.write(output_graph_def.SerializeToString())
"""

import os
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

from mayhem.utils.trainer import Trainer
from mayhem.models.lenet_model import LeNetModel

flags = flags_lib
FLAGS = flags.FLAGS

flags.DEFINE_string("logdir", "/tmp", """Where to put the log directory""")
flags.DEFINE_boolean("train", False, """Whether to retrain the model""")

def train_and_save_model(logdir):
    mnist = input_data.read_data_sets('/tmp/MNIST_data')
    model = LeNetModel()
    trainer = Trainer(model, mnist, logdir)
    trainer.train()

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
  if FLAGS.train:
    train_and_save_model(FLAGS.logdir) 

  # load_and_freeze_graph()

if __name__ == '__main__':
  tf.app.run()
