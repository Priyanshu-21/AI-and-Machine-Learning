import tensorflow as tf
import keras 
import numpy as np 

class Layer(keras.layers.Layer):
    def __init__(self, unit= 32, dim= 32):
        super().__init__()
        self.w = self.add_weight(shape= (dim, unit), initializer= 'normal', trainable= True)
        self.b = self.add_weight(shape= (unit, ), initializer= 'zeros', trainable= True)

    def call(self, inputs):
        
        return tf.matmul(inputs, self.w) + self.b

inputs = tf.ones([2, 2])
layer_input = Layer(4, inputs.ndim)
result = layer_input(inputs)

print(result)

# Assertation of weights 
assert layer_input.weights == [layer_input.w, layer_input.b]

class LayerClass(keras.layers.Layer):
    def __init__(self, dim= 32, unit= 32):

        super().__init__()
        self.total = self.add_weight(shape= (dim, ), initializer= 'zeros', trainable= False)
    
    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis= 0))

        return self.total

inputs2 = tf.ones([2, 2])
layer = LayerClass(inputs2.ndim)
result2 = layer(inputs2)

print(result2.numpy())
'''
Layer Class (Abstraction): - 
1. Weight initialization: - Initialize weights and bias of model 
2. Feed Forward: - Compute feed forward of model weight 

Layer Class (Non-trainable weights)
1. Weights & Bias: - Non - trainable and should not be used in backpropagation
'''