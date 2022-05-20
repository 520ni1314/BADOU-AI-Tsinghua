# -*- coding:utf-8 -*-
# author: Damion
# email: 1633245455@qq.com
# creation time: 2022/5/11

import numpy as np
import assignment_utils
import cv2
from keras import backend as K
from assignment_AlexNet import AlexNet
K.image_data_format() == 'channels_first'

if __name__ == '__main__':
    model = AlexNet()

    model.load_weights("./logs/last1.h5")

    img = cv2.imread("./test3.jpg")
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_nor = img_RGB / 255
    img_nor = np.expand_dims(img_nor, axis=0)
    img_resize = assignment_utils.resize_image(img_nor, (224, 224))
    # 输出分类结果，注意model.predict()与model.predict_classes()的区别
    print('the answer is: ', assignment_utils.print_answer(np.argmax(model.predict(img_resize))))

    cv2.imshow("ooo", img)
    cv2.waitKey(0)