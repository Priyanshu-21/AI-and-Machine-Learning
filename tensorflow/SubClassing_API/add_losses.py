import tensorflow as tf 
import keras 
import numpy as np 

class LossesLayer(keras.layers.Layer):
    def __init__(self, rate= 1e-3):
        super().__init__()
        self.rate = rate

    def call(self, inputs):
        self.add_loss(self.rate * tf.reduce_mean(inputs))
        return inputs

activity_loss = LossesLayer()
# layer.losses --> to get previous layer loss info during training 
assert len(activity_loss.losses) == 0   # No, layer has been called 

outputs = activity_loss(tf.zeros([2, 2]))
print(outputs)

# layer.losses == 1, as losses was calculated after layer call 
assert len(activity_loss.losses) == 1

# layer.losses == 1, as losses will be resetted 
_ = activity_loss(tf.zeros([3, 3]))
assert len(activity_loss.losses) == 1


class ActivityLoss(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.activity_loss = LossesLayer()
    
    def call(self, inputs):
        return self.activity_loss(inputs)

activity = ActivityLoss()
output1 = activity(tf.zeros([2, 2]))
print(output1)

# Layer with one Dense layer and loss calculation 
class DenseLayer(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense = keras.layers.Dense(
            32, 
            kernel_regularizer= keras.regularizers.l2(1e-3)
        )
    
    def call(self, inputs):
        return self.dense(inputs)

dense_layer = DenseLayer()
output2 = dense_layer(tf.ones([2, 2]))
print('Layer Output: ', output2)
print(dense_layer.losses)

# Model with fit method 
model = keras.Model(input= dense_layer(tf.zeros([2, 2])), output= output2)

model.compile(
    optimizer= keras.optimizers.SGD(learning_rate= 1e-3),
    loss= 'mse'
)

model.fit(np.random.uniform((2, 2)), np.random.random((2, 2)))

'''
add_loss: - calculate loss of each layer in training method 
Training Method: - Can be layer specific and user specific (Training Loops) 
'''