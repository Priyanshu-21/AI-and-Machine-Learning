import tensorflow as tf 
import keras 
import numpy as np

class InitialWeights(keras.layers.Layer):
    def __init__(self, unit= 32):
        super().__init__()
        self.unit = unit
    
    def build(self, dims):
        self.w = self.add_weight(shape= (dims[-1], self.unit), initializer= 'random_normal', trainable= True)
        self.b = self.add_weight(shape= (self.unit, ), initializer= 'zeros', trainable= True)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


initial_weight = InitialWeights(4)
inputs = tf.ones([2, 2])
result = initial_weight(inputs)

result1 = initial_weight(inputs)
print('Result 1: ', result)
print('\n')
print('Result 2: ', result1)

class ReusableLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Define sub-layer structure 
        self.layer_1 = InitialWeights(32)
        self.layer_2 = InitialWeights(32)
        self.layer_3 = InitialWeights(1)
    
    def call(self, inputs):
        x = self.layer_1(inputs)
        x = tf.nn.relu(x)
        x = self.layer_2(x)
        x = tf.nn.relu(x)
        x = self.layer_3(x)

        return x
    

reusable_layer = ReusableLayer()
inputs = tf.ones(shape= (10, 8))
y = reusable_layer(inputs)

print("Reusable Layer: ", y)
print('\n')
print('Weights: ', len(reusable_layer.weights))
print('Trainable Weights: ', len(reusable_layer.trainable_weights))

'''
Implementation: - assign & initialize weights in different method 
Use: - Each instance (obj) should have weights initialization 
Imp: - Each object is having same initial weights and bias values to be used in layers. 
'''