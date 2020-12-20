import cv2
import os
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image
img=cv2.imread("realtest/MRI2.jpg") 

image_from_array = Image.fromarray(img, 'L')

image_from_array.show()
size_image = image_from_array.resize((224,224))
p = np.expand_dims(size_image, 0)
img = tf.cast(p, tf.float32)
saved_model = load_model("my_model_weights.h5")
print(['Infected','Uninfected'][np.argmax(saved_model.predict(img))])