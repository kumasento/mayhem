#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import app
from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import gfile 

flags = flags_lib
FLAGS = flags.FLAGS

flags.DEFINE_string("input", "", """TensorFlow 'GraphDef' file to load.""")

def main(unused_args):
  if not gfile.Exists(FLAGS.input):
    print("Input graph file '" + FLAGS.input + "' does not exist!")
    return -1

  # Save and load GraphDef from local files
  tf_graph = graph_pb2.GraphDef()
  with gfile.Open(FLAGS.input, "rb") as f:
    data = f.read()
    tf_graph.ParseFromString(data)

    for node in tf_graph.node:
      print(node.name)

  graph = ops.Graph()
  with graph.as_default():
    importer.import_graph_def(tf_graph, input_map={}, name="")


if __name__ == '__main__':
    app.run()
