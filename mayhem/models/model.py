"""
THIS FILE HAS BEEN DEPRECATED!

We currently don't want to use the ABC.
"""
from abc import ABCMeta, abstractmethod

class Model(metaclass=ABCMeta):
    """
    An abstract base class for the model classes defined
    in TensorFlow.
    """

    @abstractmethod
    def inference(self):
        
        pass

    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluation(self):
        pass
