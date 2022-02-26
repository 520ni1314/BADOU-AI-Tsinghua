import cv2
import random
from numpy import *
def PepperandSalt(src,percetage):
    NoiseImg=src
    NoiseNum=int(percetage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.random_integers(0,src.shape[0]-1)
        randY=random.random_integers(0,src.shape[1]-1)
        if random.random_integers(0,1)<=0.5:
            NoiseImg[randX,randY]=0
        else:
            NoiseImg[randX,randY]=255
    return NoiseImg

img=cv2.imread('lenna.png',0)
cv2.imshow("original", img)
img1=PepperandSalt(img,0.2)
cv2.imshow('lenna_PepperandSalt',img1)
cv2.waitKey(0)

