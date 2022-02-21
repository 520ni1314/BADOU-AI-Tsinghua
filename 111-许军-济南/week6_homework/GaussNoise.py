# -- coding:utf-8 --
import random
import cv2
import matplotlib.pyplot as plt
def GaussNoise(img,mean,sigma,percetage):
    noisenum = int(percetage * img.shape[0] * img.shape[1])
    for i in range(noisenum):
        randx=random.randint(0, img.shape[0] - 1)
        randy=random.randint(0, img.shape[1] - 1)
        img[randx][randy] = img[randx][randy] + random.gauss(mean,sigma)
        if img[randx][randy] > 255:
            img[randx][randy] = 255
        if img[randx][randy] < 0:
            img[randx][randy] = 0
    return img

if __name__ == '__main__':
    img = cv2.imread("./img/lenna.png",0)
    img2 = GaussNoise(img,2,6,0.8)
    plt.subplot(1,2,1)
    plt.imshow(img,cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(img2,cmap="gray")
    plt.show()

