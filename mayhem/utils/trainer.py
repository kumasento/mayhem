#!/usr/bin/env python

import os
import sys
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

class Trainer(object):
    """
    Trainer wraps the training process based on TensorFlow.
    
    It have the following objectives:
    - With model and dataset provided, with a simple function call
    we could run the whole training process.
    - We could easily export and freeze a model based on the trained
    result.
    """

    def __init__(self, model, dataset, logdir='/tmp'):
        self._model   = model
        self._dataset = dataset
        self._logdir  = logdir

        if not os.path.isdir(self._logdir):
            os.mkdir(self._logdir)

    def train(self):
        graph = tf.Graph()

        logits, loss, train_op, evaluation = self._model.build(graph)

        with graph.as_default():
            saver = tf.train.Saver()

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for i in range(2000):
                    batch = self._dataset.train.next_batch(50)

                    if i % 100 == 0:
                        print('Model saved in file %s' %
                                saver.save(
                                    sess,
                                    self._logdir + '/' + self._model.name,
                                    global_step=i))

                    sess.run(train_op, feed_dict={ 
                        self._model.input_placeholder: batch[0],
                        self._model.label_placeholder: batch[1]
                    })

                print('Test accuracy: %g' % 
                        sess.run(
                            evaluation,
                            feed_dict={
                                self._model.input_placeholder: self._dataset.test.images,
                                self._model.label_placeholder: self._dataset.test.labels
                            }))

                output_graph_def = graph_util.convert_variables_to_constants(
                        sess,
                        graph.as_graph_def(),
                        [ logits.name[:-2] ])
                with gfile.GFile(self._logdir + '/' + self._model.name + '.pb', 'wb') as f:
                    f.write(output_graph_def.SerializeToString())

        return None
