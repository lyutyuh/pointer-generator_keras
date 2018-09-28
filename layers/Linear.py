# -*- coding=utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from keras.engine import Layer
import keras.backend as K
from keras import activations
from keras import initializers

class Linear(Layer):
    def __init__(self, output_size, bias, bias_start=0.0, activation=None):
        self.output_size = output_size
        self.bias = bias
        self.bias_start = bias_start
        self.activation = None
        if activation is not None:
            self.activation = activations.get(activation)
        self.matrix = None
        super(Linear, self).__init__()

    def build(self, input_shape):
        assert isinstance(input_shape, list), "The input of linear layer must be a list of tensors!"
        self.total_arg_size = 0
        self.in_shape = input_shape
        shapes = [a for a in input_shape]
        for shape in shapes:
            if len(shape) != 2:
                raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
            if not shape[1]:
                raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
            else:
                self.total_arg_size += shape[1]

        self.matrix = self.add_weight(shape=(self.total_arg_size, self.output_size),
                                      initializer="glorot_uniform",
                                      name='kernel',
                                      regularizer=None,
                                      constraint=None)

        if self.bias:
            self.bias = self.add_weight(shape=(self.output_size,),
                                        initializer=initializers.Constant(value=self.bias_start),
                                        name='bias',
                                        regularizer=None,
                                        constraint=None)
        super(Linear, self).build(input_shape)

    def call(self, inputs):
        # Now the computation.
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        if len(inputs) == 1:
            res = K.dot(inputs[0], self.matrix)
        else:
            res = K.dot(K.concatenate(inputs, axis=1), self.matrix)
        if not self.bias:
            return res
        res = K.bias_add(res, self.bias, data_format='channels_last')
        if self.activation is not None:
            res = self.activation(res)
        return res

    def compute_output_shape(self, input_shape):
        output_shape = list(self.in_shape[0])
        output_shape[-1] = self.output_size
        return tuple(output_shape)
