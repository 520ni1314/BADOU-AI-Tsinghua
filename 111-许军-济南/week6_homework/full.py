# -- coding:utf-8 --
import cv2
import random
import matplotlib.pyplot as plt
def full1(img,percetage):
    img2 = img
    noisenum=int(percetage * img2.shape[0] * img2.shape[1])
    for i in range(noisenum) :
        randx=random.randint(0, img2.shape[0] - 1)
        randy=random.randint(0, img2.shape[1] - 1)
        if random.random() > 0.5 :
            img2[randx][randy]=255
        if random.random() < 0.5 :
            img2[randx][randy] = 0
    return img2
if __name__ == '__main__':
    img = cv2.imread("./img/lenna.png",0)
    img2 = full1(img,0.2)
    plt.subplot(1,2,1)
    plt.imshow(img,cmap="gray")
    plt.show()
