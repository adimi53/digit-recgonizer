import cv2
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import os

mnist = tf.keras.datasets.mnist     #load the dataset from tenserflow
(x_train, y_train) , (x_test, y_test) =mnist.load_data()   #Split the labeled data to Test and Traingng #x for image, y for classification (which digit is it really)

#normalize values of grayscale from 0-255 to 0-1
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

#setup the model
model = tf.keras.models.Sequential()
#adding layers
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #input layer. flatten converts matrics in size of x*x to x^2*1 
model.add(tf.keras.layers.Dense(1028,activation='relu')) #add hidden layer with 128 neuorns (REMINDER: check what 'relu' means)
model.add(tf.keras.layers.Dense(1028,activation='relu')) #add hidden layer with 128 neuorns(REMINDER: check what 'relu' means)
model.add(tf.keras.layers.Dense(10,activation='softmax')) #add hidden layer with 128 neuorns(REMINDER: check what 'softmax' means)

#compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy' ,metrics=['accuracy'])

#train model
model.fit(x_train,y_train,epochs=10)

loss,accuracy = model.evaluate(x_test , y_test) 
print (loss)
print(accuracy)

model.save('handwritten_digits_model.keras')


