import cv2
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import os
"""""
mnist = tf.keras.datasets.mnist     #load the dataset from tenserflow
(x_train, y_train) , (x_test, y_test) =mnist.load_data()   #Split the labeled data to Test and Traingng #x for image, y for classification (which digit is it really)

#normalize values of grayscale from 0-255 to 0-1
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)
"""

#load trained model
model= tf.keras.models.load_model('handwritten_digits_model.keras')

index = 1
while os.path.isfile(f"digits/{index}.png"):
    try:
        img = cv2.imread(f"digits/{index}.png", cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
        img = np.invert(np.array([img]))  # Invert image
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)
        print(f"Digit prediction: {predicted_digit}")

        # Display the image along with the prediction
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.title(f"Predicted digit: {predicted_digit}")
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        index += 1