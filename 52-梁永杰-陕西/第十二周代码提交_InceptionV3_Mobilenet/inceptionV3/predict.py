import tensorflow as tf
from model.InceptionV3 import InceptionV3
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.imagenet_utils import decode_predictions

def preprocess_input(x):
    '''
    对数据进行预处理
    :param x:
    :return:
    '''
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


model = InceptionV3()
path = './logs/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
model.load_weights(path)
img_path = 'elephant.jpg'
img = image.load_img(img_path,target_size=(299,299))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)  # 在图片第一维度前增加一个维度

x = preprocess_input(x)

preds = model.predict(x)
print('Predicted',decode_predictions(preds))

