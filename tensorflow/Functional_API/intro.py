# Introduction to creating model using functional API 
import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from keras import layers


# Model Architecture 
class SimpleConvolutional():
    def __init__(self):
        self.inputs = keras.Input(shape= (784, ))
        self.layer1 = keras.layers.Dense(64, activation= 'relu')
        self.layer2 = keras.layers.Dense(32, activation= 'relu')
        self.out = keras.layers.Dense(10)

    def call(self):
        input = self.inputs
        x = self.layer1(input)
        x = self.layer2(x)
        outputs = self.out(x) 
        
        return outputs


simple_model = SimpleConvolutional()
outputs = simple_model.call()
model = keras.Model(inputs= simple_model.inputs, outputs= outputs, name= 'simple_model', show_shapes= True)

# Data-Collection 
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Data preprocessing and normalization 
x_train = x_train.reshape(x_train.shape[0], 784).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 784).astype('float32')

# Normalization 
x_train = x_train / 255
x_test = x_test / 255

# Model Training and Evaluation 
model.compile(
    optimizer= keras.optimizers.RMSprop(),
    loss= keras.losses.SparseCategoricalCrossentropy(from_logits= True),
    metrics= [keras.metrics.SparseCategoricalAccuracy()] 
)

# Model Training 
history = model.fit(x_train, y_train, validation_data= (x_test, y_test), batch_size= 64, epochs= 10, verbose= 2)

# Model evaluation
evaul = model.evaluate(x_test, y_test, verbose= 2)

print('Model loss: ', evaul[0])
print('Model Accuracy: ', evaul[1])

# Save Model 
model.save('simple_cnn.keras')
