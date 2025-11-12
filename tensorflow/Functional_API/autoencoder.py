# Implementation of auto-encoder using functional API
import tensorflow as tf 
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt 

class AutoEncoder():
    def __init__(self, features= 28):
        self.inputs = layers.Input(shape= (features, features, 1))

        # Encoder layers
        self.encode1 = layers.Conv2D(64, 3, activation= 'relu', padding= 'same')
        self.encode2 = layers.Conv2D(16, 2, activation= 'relu', padding= 'same')
        self.pool = layers.MaxPooling2D(3)
        self.encoder_output = layers.GlobalMaxPool2D()

        # Decoder layers 
        self.decoder_inputs = layers.Reshape(target_shape= (4, 4, 1))
        self.decode1 = layers.Conv2DTranspose(32, 2, activation= 'relu', padding= 'same')
        self.decode2 = layers.Conv2DTranspose(64, 3, activation= 'relu', padding= 'same')
        self.depool = layers.UpSampling2D(size= (7, 7))
        self.corrected_layer = layers.Conv2DTranspose(16, 2, activation= 'relu', padding= 'same')
        self.decoder_output = layers.Conv2DTranspose(1, 3, activation= 'sigmoid', padding= 'same')

    def call(self, inputs):
    
        # Encoder Layer
        x = self.encode1(inputs)
        x = self.encode2(x)
        x = self.pool(x)
        x = self.encoder_output(x)
        # Decoder layer
        y = self.decoder_inputs(x)
        y = self.decode1(y)
        y = self.decode2(y)
        y = self.depool(y)
        y = self.corrected_layer(y)
        outputs = self.decoder_output(y)

        return outputs

# Data - Collection 
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

# Data Pre-processing (Normalization)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Expand Dimensions: - (batch_size, 28, 28, 1)
x_train = np.expand_dims(x_train, -1) 
x_test = np.expand_dims(x_test, -1)

autoencoder = AutoEncoder()
inputs = autoencoder.inputs
outputs = autoencoder.call(inputs)

model = keras.Model(inputs = inputs, outputs= outputs, name= 'AutoEncoder')

# Model compile 
model.compile(
    optimizer= 'adam', 
    loss= 'mse'
)

# Model Summary
model.summary()

# Model Training 
history = model.fit(x_train, x_train, validation_data= (x_test, x_test), batch_size= 128, epochs= 10, verbose= 2)

# Model Testing 
decoded_img = model.predict(x_test[:10]) # pick up the first 10 images to predict

# Image Visualization 
n = 10
plt.figure(figsize= (20, 4))

for i in range(n):

    # Original Image 
    plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap= 'gray')
    plt.title('Original Image')
    plt.axis(False)

    # Reconstructed Image
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_img[i].reshape(28, 28), cmap= 'gray')
    plt.title('Reconstructed Image')
    plt.axis(False)


plt.show()

#TODO: - Model is trainable but not generating image output