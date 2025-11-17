import tensorflow as tf 
import numpy as np 
import keras 
from keras import layers

class Sampling(layers.Layer):
    # Vector encoding a digit, (z_mean, z_var_log) ==> z 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        z_mean, z_var_log = inputs
        batch = tf.shape(z_mean)[0]
        dims = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape= (batch, dims))
        
        return z_mean + tf.exp(0.5 * z_var_log) * epsilon

class Encoder(layers.Layer):
    # Encoding an image into probablity distribution (z_mean, z_var_log, z)
    def __init__(self, latent_dims= 32, intermediate_dims= 64, name= 'encoder', **kwargs):
        super().__init__(name= name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dims, activation= 'relu')
        self.dense_mean = layers.Dense(latent_dims)
        self.dense_var_log = layers.Dense(latent_dims)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_var_log = self.dense_var_log(x)
        z = self.sampling((z_mean, z_var_log))

        return z_mean, z_var_log, z


# Sample Input 
encoder = Encoder()
inputs = tf.ones([3, 784])
outputs = encoder(inputs)

print('Encoder Output: ', outputs)
'''
Implementation of variational autoEncoder: Generative Predictive Modeling 
1. Encoder Layer: - (batch_size, model_dims, intermediate_dims) ==> (z_mean, z_var_loss, z)
2. Sampling Layer: - (Sampling(z_mean + exp(0.5 * var_log) * random_normal(batch, dims)
'''