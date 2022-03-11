import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from model.mobilenet import Mobilenet

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

model = Mobilenet()

weigth_path = './logs/mobilenet_1_0_224_tf.h5'
model.load_weights(weigth_path)

img_path = 'elephant.jpg'
img = image.load_img(img_path,target_size=(224,224))
x = image.img_to_array(img)

x = np.expand_dims(x,axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print(np.argmax(preds))
print('Predicted',decode_predictions(preds,1)) #
