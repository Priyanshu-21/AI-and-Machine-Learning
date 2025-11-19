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


class Decoder(layers.Layer):
    # Decoding z and re-constructing z back to image 
    def __init__(self, original_dims, intermediate_dims= 64, name='decoder', **kwargs):
        super().__init__(name= name, **kwargs)
        self.proj = layers.Dense(intermediate_dims, activation= 'relu')
        self.outputs = layers.Dense(original_dims, activation= 'sigmoid')
    
    def call(self, inputs):
        x = self.proj(inputs)
        return self.outputs(x)


class AutoEncoder(keras.Model):
    def __init__(
        self, 
        original_dims, 
        intermediate_dims = 64, 
        latent_dims = 32, 
        name = "AutoEncoder",
        **kwargs
    ):
        super().__init__(name= name, **kwargs)
        self.encoder = Encoder(latent_dims, intermediate_dims)
        self.decoder = Decoder(original_dims, intermediate_dims)
    
    def call(self, inputs):
        z_mean, z_var, z = self.encoder(inputs)
        reconstructed_img = self.decoder(z)
        
        # add loss method (KL Divergence Regularization Loss)
        kl_loss = -0.5 * tf.reduce_mean(
            z_var - tf.square(z_mean) - tf.exp(z_var) + 1
        )

        self.add_loss(kl_loss)
        return reconstructed_img


original_dims = 784
vae_autoencoder = AutoEncoder(original_dims)

optimizers = keras.optimizers.Adam(learning_rate= 1e-3)
mse_loss_fn = keras.losses.MeanSquaredError()
loss_metrics = keras.metrics.Mean()

# Data Collection (Source: mnist)
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Data pre-processing (flatten image)
x_train = x_train.reshape(x_train.shape[0], 784).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 784).astype('float32') / 255

# x_train dataset -- Shuffle in batch 64 
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size= 1024).batch(64)

# Training Loop for 5 Epochs 
epochs = 5

for epoch in range(epochs):
    print('Number of Epochs of training: ', epoch)

    for step, train_batch in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed_image = vae_autoencoder(train_batch)

            # Loss Calculation:
            loss = mse_loss_fn(train_batch, reconstructed_image)
            # KLD regualrization loss 
            loss += sum(vae_autoencoder.losses)
        
        grads = tape.gradient(loss, vae_autoencoder.trainable_weights)
        optimizers.apply_gradients(zip(grads, vae_autoencoder.trainable_weights))

        loss_metrics(loss)

        if (step % 100) == 0:
            print('Mean Loss in step %d is %.4f' % (step, loss_metrics.result()))

# TODO: - Need to understand model evaluation & prediction 
'''
# Sample Input 
auto_encoder = AutoEncoder(784)
inputs = tf.ones([3, 784])

outputs = auto_encoder(inputs)

print('Original Inputs: ', inputs)
print('\n')
print('AutoEncoded Outputs: ', outputs)
'''

'''
Implementation of variational autoEncoder: Generative Predictive Modeling 
1. Encoder Layer: - (batch_size, model_dims, intermediate_dims) ==> (z_mean, z_var_loss, z)
2. Sampling Layer: - (Sampling(z_mean + exp(0.5 * var_log) * random_normal(batch, dims)
3. Decoder Layer: - Reconstruct the image from the z - space of the resulted Encoder layer 

Model (AutoEncoder): - 
1. Call -> Encoder -> Sampling -> Decoder -> Reconstructed image 
'''