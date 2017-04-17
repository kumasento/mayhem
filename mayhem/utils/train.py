#!/usr/bin/env python

## @package 

import tensorflow as tf

class Trainer(object):
    def __init__(self, model, dataset, logdir='/tmp'):
        self._model = model
        self._dataset = dataset
        self._logdir = logdir

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
                                saver.save(sess, self._logdir))

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

        return None
