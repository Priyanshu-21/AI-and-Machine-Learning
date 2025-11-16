import tensorflow as tf 
import keras 
import numpy as np 

class Layer(keras.layers.Layer):
    def __init__(self, units= 32):
        super().__init__()
        self.units = units 
    
    def build(self, input_dims):
        self.w = self.add_weight(
            shape= (input_dims[-1], self.units), 
            initializer= 'random_normal', 
            trainable= True
        )
        self.b = self.add_weight(
            shape= (self.units, ), 
            initializer= 'random_normal', 
            trainable= True
        )
    
    def call(self, inputs):
        return tf.matmul(self.w, inputs) + self.b

    def get_config(self):
        return {'units': self.units}


input_layer = Layer(64)
config = input_layer.get_config()
# Due to super() in init method, we can use method of base class (Layer)
new_layer = input_layer.from_config(config)

print('Input layer: ', input_layer)
print('New Layer: ', new_layer)

# Class Implementation for Serialized and De-Serialized layer
class SerializationLayer(keras.layers.Layer):
    def __init__(self, units= 32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    
    def build(self, input_dims):
        self.w = self.add_weight(
            shape= (input_dims[-1], self.units),
            initializer= 'random_normal',
            trainable= True,
        )
        self.b = self.add_weight(
            shape= (self.units, ),
            initializer= 'random_normal',
            trainable= True,
        )

    # Serialization
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, inputs):
        return tf.matmul(self.w, inputs) + self.b
    

layer = SerializationLayer(64)

config = layer.get_config()
print('Layer Config: ', config)
new_layer = SerializationLayer.from_config(config)

print('Layer 1: ', layer)
print("New Layer Information: ", new_layer)

assert type(layer == new_layer)

'''
Serializaiton: - Introduction to serialization 
1. Used to select the same layer configuration for multiple layer creation 
2. Used to serialize and de-serialize the layer
'''