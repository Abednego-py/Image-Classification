import keras
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


class_names = ['aeroplane', 'car', 'bird']

def teachable_machine_classification(img, model):
    # Load the model
    # model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 32, 32, 3), dtype=np.float32)
    image = img
    #image sizing
    size = (32, 32)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    #turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = image_array.astype(np.float32) / 255

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    pred = model.predict(data)
    # col = 'green' if np.argmax(y[i])==np.argmax(pred[i]) else 'red'
    return np.argmax(pred[0])
    # return np.argmax(prediction) # return position of the highest probability