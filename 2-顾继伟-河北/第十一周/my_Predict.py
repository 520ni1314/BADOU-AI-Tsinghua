import numpy as np
import cv2
import my_Utils
from keras import backend as K
import my_AlexNet
K.image_data_format()=="channel_first"

if __name__=='__main__':
    model = my_AlexNet.my_AlexNet()
    model.load_weights("./logs/last1.h5")
    img = cv2.imread("./test4.jpg")
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgarr = imgRGB/255
    imgarr = np.expand_dims(imgarr,axis=0)
    img_resize = my_Utils.resize_image(imgarr,(224,224))

    print("result is:",my_Utils.print_result(np.argmax(model.predict(img_resize))))

    cv2.imshow("测试图像",img)
    cv2.waitKey(0)
