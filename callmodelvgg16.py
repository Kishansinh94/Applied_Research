import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
img = image.load_img("test/no/no.jpg",target_size=(224,224), grayscale=True)
#img = np.asarray(img)

#used this new line instead of old one above by time
img=np.reshape(img,(224,224,1))


img = np.expand_dims(img, axis=0)
from keras.models import load_model
#saved_model = load_model("vgg16.h5")
saved_model = load_model("my_model_weights.h5")
output = saved_model.predict(img)
if output[0][0] > output[0][1]:
    print("No")
else:
    print('YES')


