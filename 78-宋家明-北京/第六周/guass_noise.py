import numpy as np
import cv2
import random

def GaussianNoise(src,means,sigma,percetage):
    noiseimg = src
    h, w = src.shape[0],src.shape[1]
    assert percetage<=1
    nois_h, nois_w = percetage*h, percetage*w
    left, right = int((h-nois_h)//2), int((h-nois_h)//2+nois_h)
    up, down = int((w-nois_w)//2), int((w-nois_w)//2+nois_w)
    gaussian_array = np.random.normal(means,sigma,(right-left,down-up))
    noiseimg[up:down,left:right] = noiseimg[up:down,left:right] + gaussian_array
    noiseimg = noiseimg.astype(np.uint8)
    return noiseimg


if __name__=='__main__':

    print('Gaussian Noise img')
    img = cv2.imread('../lenna.png')
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    noiseimg = GaussianNoise(gray_img,2,0.4,0.7)
    print(noiseimg)
    print(noiseimg.shape)
    print(gray_img.shape)
    cv2.imshow('gaussianimg',noiseimg)
    cv2.imshow('img',gray_img)
    cv2.waitKey(0)

