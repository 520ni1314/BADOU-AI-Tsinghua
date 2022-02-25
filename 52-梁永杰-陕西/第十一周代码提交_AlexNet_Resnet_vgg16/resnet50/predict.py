import numpy as np
import cv2
import utils
from tensorflow.keras import backend as K
from model.resnet50 import Resnet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

K.image_data_format() == 'channels_first'

if __name__ == '__main__':
    Resnet50 =Resnet50()

    layer_fc1 = Dense(1024,activation='relu',name='fc1024')(Resnet50.layers[-2].output)
    layer_fc2 = Dense(2, activation='softmax', name='fc2')(layer_fc1)
    model = Model(Resnet50.input, layer_fc2,name='change_Resnet50')
    model.summary()

    weight_path = './logs/last1.h5'
    model.load_weights(weight_path)


    img = cv2.imread('test.jpg')
    img_RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_nor = img_RGB/255

    img_nor = np.expand_dims(img_nor,axis=0)  # 增加数据维度，与模型输入形状一直
    img_resize = utils.resize_image(img_nor, (224, 224))

    pred_data = model.predict(img_resize)
    print(pred_data)
    print("预测图片的结果为：",utils.print_answer(np.argmax(pred_data)))