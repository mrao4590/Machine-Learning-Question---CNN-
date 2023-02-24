from __future__ import division,print_function, absolute_import
from flask import Flask, render_template, request
from tensorflow import keras
from PIL import Image

import numpy as np

from keras.utils import to_categorical
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.models import model_from_json
import os

app = Flask(__name__)


# Define the home page route
@app.route('/')
def home():
    return render_template('home.html')

def ml_model(img):
	
	mnist = keras.datasets.mnist
	(x_train,y_train) , (x_test , y_test) = mnist.load_data()
	x_train, x_test = x_train/225.0, x_test/225.0
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	X_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
	input_shape = (28, 28, 1)
	y_test , y_train = to_categorical(y_test,10), to_categorical(y_train,10)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train.shape[1:]
	model = tf.keras.models.Sequential([
	#1st layer
	# Conv2D( number_of_filters , kernal_size , input_shape(add this parameter just for the input conv layer))
	tf.keras.layers.Conv2D(64,(3,3),input_shape=x_train.shape[1:],activation='relu'),
	tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
	# defining the pooling for this layer
	tf.keras.layers.MaxPooling2D(pool_size= (2,2)),
	#2nd layer
	tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
	# defining the pooling for this layer
	tf.keras.layers.MaxPooling2D(pool_size= (2,2)),

	#FC layers
	tf.keras.layers.Dropout(0.3),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation = 'relu'),
	tf.keras.layers.Dropout(0.1),
	tf.keras.layers.Dense(10,activation = 'softmax')
	])

	model.compile(optimizer = "adam",loss = "categorical_crossentropy",metrics = ['accuracy'])
	model.fit(x_train, y_train, batch_size = 256,epochs = 7,verbose = 1)

	pred = model.predict(img)
	d= np.argmax(pred)
	return d
# Define the route for processing the uploaded image
@app.route('/predict', methods=['POST'])
def predict():
    # Preprocess the image
    img = Image.open(request.files['image'])
    img = img.resize((28, 28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img / 255.0
    
    # Make the prediction
    digit =ml_model(img)
    #digit = np.argmax(prediction)
    
    # Display the predicted digit
    return render_template('result.html', prediction=digit)

if __name__ == '__main__':
    app.run(debug=True)
