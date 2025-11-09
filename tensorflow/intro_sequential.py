# Introduction to sequential model in keras 
import tensorflow as tf
from keras.layers import Input, Dense

def my_model():

    model = tf.keras.models.Sequential([
        Input(shape= (64, )),
        Dense(128, activation= 'relu'),
        Dense(64, activation= 'relu'),
        Dense(4, activation= 'softmax'),
    ])

    return model 

model = my_model()

print(model)
