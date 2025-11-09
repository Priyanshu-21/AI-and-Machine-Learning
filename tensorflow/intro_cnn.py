import tensorflow as tf
from keras import layers, Model, datasets
import matplotlib.pyplot as plt 

class ConvolutionalModel(Model):
    
    def __init__(self, input_shape: tuple, n_classes: int):
        super().__init__()

        # Convolutional layer 1
        self.convlayer1 = tf.keras.Sequential([
            layers.Conv2D(64, kernel_size= [4, 4], strides= (2, 2), activation= 'relu'),
            layers.MaxPooling2D(pool_size= (4, 4), strides= (2, 2))
        ])

        # Convolutional layer 2
        self.convlayer2 = tf.keras.Sequential([
            layers.Conv2D(32, kernel_size= [2, 2], strides= (2, 2), activation= 'relu'),
            layers.MaxPooling2D(pool_size= (2, 2), strides= (1, 1))
        ])

        #Fully-Connected layer
        self.connected_layer = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(100, activation= 'relu'),
        ])

        #Output-Layer
        self.classifier = layers.Dense(n_classes, activation= 'softmax')
    
    def call(self, inputs):
        x = self.convlayer1(inputs)
        x = self.convlayer2(x)
        x = self.connected_layer(x)

        return self.classifier(x)


# Data Collection 
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Feature Extraction (Re-Shaping, Normalization)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

# One-Hot Encoding for categorial features 
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Explainatory columuns and features 
nfeatures = x_train.shape[1]
nclasses = y_train.shape[1]


model = ConvolutionalModel(input_shape=(nfeatures, nfeatures, 1), n_classes=nclasses)
model.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics=['accuracy'])

# Model Training 
model_training = model.fit(x_train, y_train, validation_data= [x_test, y_test], batch_size= 200, epochs= 10, verbose= 2)


# Model Evaluation 
test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size= 200)

#Model Visualization 
plt.figure(figsize= (8, 8))
plt.plot(model_training.history['accuracy'], label= 'Training', marker = '-')
plt.plot(model_training.history['val_accuracy'], label= 'Val Accuracy', marker= '-')
plt.xlabel('Epochs')
plt.ylabel('Accuracy & Loss')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.show()